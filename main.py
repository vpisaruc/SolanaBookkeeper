import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
import numpy as np
import schedule

# Write your wallet address
WALLET_ADDRESS = "YOUR_WALLET_ADRESS"

TELEGRAM_BOT_TOKEN = "7881914890:AAH6HMFK7HvYER-xI_5vVwgfr9F6gpHNBtc"
# Write your TELEGRAM chat id
TELEGRAM_CHAT_ID = "YOUR_TG_CHAT_ID"

HELIUS_API_KEY = "YOUR__HELIUSNODE_API_KEY"
HELIUS_RPC_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

# create your google_cloud_token, instruction in read.me
SERVICE_ACCOUNT_FILE = "g_cloud_toke.json"
# Write your spreadsheet name 
SPREADSHEET_NAME = "YOUR_SPREADSHEET_NAME"
# Write your sheet name 
SHEET_NAME = "YOUR_SHEET_NAME"
# Your Google spreadsheet link 
SPREADSHEET_LINK = "https://docs.google.com/spreadsheets/d/1F3C7zeSu5v87rcfwmJiQRGibMEgRsA9EyP17XY7yJRI/edit?gid=1974413569#gid=1974413569"


def unix_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def get_signatures_for_wallet(wallet_address, start_date, end_date, limit=1000):
    all_signatures = []
    before = None  # for pagination

    start_timestamp = int(start_date.replace(tzinfo=timezone.utc).timestamp())
    end_timestamp = int(end_date.replace(tzinfo=timezone.utc).timestamp())

    while True:
        params = {"limit": limit, "commitment": "finalized"}
        if before:
            params["before"] = before  # load next page

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [wallet_address, params]
        }

        response = requests.post(HELIUS_RPC_URL, json=payload)
        if response.status_code != 200:
            break  # if error break

        try:
            transactions = response.json().get("result", [])
        except requests.JSONDecodeError:
            print("JSON error in get_signatures_for_wallet", response.text[:500])
            break

        if not transactions:
            break  # If thre are no more transaction -> break

        # Filter transaction by timestamp
        filtered_transactions = [
            (tx["signature"], tx["slot"]) 
            for tx in transactions 
            if start_timestamp <= tx.get("blockTime", 0) <= end_timestamp
        ]
        all_signatures.extend(filtered_transactions)

        # We check that the limit is met (i.e. the API returned the maximum possible number of records)
        if len(transactions) < limit:
            break  # If you received less than the limit, then there are no further entries

        before = transactions[-1]["signature"]  # We take the last signature for the next request

    return all_signatures

def get_block(slot):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBlock",
        "params": [slot, {
            "encoding": "jsonParsed",
            "transactionDetails": "full",
            "rewards": False,
            "maxSupportedTransactionVersion": 0
        }]
    }

    response = requests.post(HELIUS_RPC_URL, json=payload)
    if response.status_code != 200:
        print(response.status_code, response.text)
        return {}

    try:
        return response.json().get("result", {})
    except requests.JSONDecodeError:
        print("JSON error in get_block", response.text[:500])
        return {}

def filter_transactions_by_wallet(block_data, wallet_address):
    transactions = block_data.get("transactions", [])

    # Filter: check that the wallet is in accountKeys and it has signer=True
    filtered = [
        tx for tx in transactions
        if any(acc.get("pubkey") == wallet_address and acc.get("signer") for acc in tx["transaction"]["message"]["accountKeys"])
    ]

    return filtered

def extract_fail_reason(transaction):
    if transaction.get("meta", {}).get("err") is None:
        return ""
    log_messages = transaction.get("meta", {}).get("logMessages", [])
    for log in log_messages:
        if "exceeds desired slippage limit" in log:
            return "Slippage"
    return "Other"

def extract_tips(transaction, wallet_address):
    instructions = transaction.get("transaction", {}).get("message", {}).get("instructions", [])
    tips_amount, tips_destination = 0, None

    for instr in instructions:
        if instr.get("program") == "system" and instr.get("parsed", {}).get("type") == "transfer":
            info = instr.get("parsed", {}).get("info", {})
            if info.get("source") == wallet_address:
                tips_amount += int(info.get("lamports", 0)) / 1e9
                tips_destination = info.get("destination")

    return round(tips_amount, 9), tips_destination

def extract_trades(transaction, wallet_address):
    pre_balances = transaction.get("meta", {}).get("preTokenBalances", [])
    post_balances = transaction.get("meta", {}).get("postTokenBalances", [])

    pre_dict = {item["mint"]: item["uiTokenAmount"]["uiAmount"] or 0 for item in pre_balances if item["owner"] == wallet_address}
    post_dict = {item["mint"]: item["uiTokenAmount"]["uiAmount"] or 0 for item in post_balances if item["owner"] == wallet_address}

    token_buy, amount_buy, token_sell, amount_sell = None, 0, None, 0
    missing_mints = set()

    for mint in pre_dict.keys() | post_dict.keys():
        pre_amount = pre_dict.get(mint)
        post_amount = post_dict.get(mint)

        if pre_amount is None or post_amount is None:
            missing_mints.add(mint)
            continue
        else:
            delta = post_amount - pre_amount

        if delta > 0:
            token_buy = "wSOL" if mint == "So11111111111111111111111111111111111111112" else mint
            amount_buy = delta
        elif delta < 0:
            token_sell = "wSOL" if mint == "So11111111111111111111111111111111111111112" else mint
            amount_sell = abs(delta)

        if delta == 0:
            missing_mints.add(mint)

    if amount_buy == 0 or amount_sell == 0:
        all_pre_balances = {item["mint"]: {} for item in pre_balances}
        all_post_balances = {item["mint"]: {} for item in post_balances}

        
        for item in pre_balances:
            all_pre_balances[item["mint"]][item["owner"]] = item["uiTokenAmount"]["uiAmount"] or 0
        for item in post_balances:
            all_post_balances[item["mint"]][item["owner"]] = item["uiTokenAmount"]["uiAmount"] or 0

        for mint in missing_mints:
            if mint in all_pre_balances and mint in all_post_balances:
                max_owner = max(
                    (owner for owner in all_pre_balances[mint] if owner != wallet_address),
                    key=lambda owner: abs((all_post_balances[mint].get(owner, 0) or 0) - (all_pre_balances[mint].get(owner, 0) or 0)),
                    default=None
                )
                
                if max_owner and max_owner in all_post_balances[mint]:
                    pre_amount = all_pre_balances[mint].get(max_owner, 0) or 0
                    post_amount = all_post_balances[mint].get(max_owner, 0) or 0
                    delta = pre_amount - post_amount 

                    if delta > 0 and amount_buy == 0:
                        token_buy = "wSOL" if mint == "So11111111111111111111111111111111111111112" else mint
                        amount_buy = delta
                    elif delta < 0 and amount_sell == 0:
                        token_sell = "wSOL" if mint == "So11111111111111111111111111111111111111112" else mint
                        amount_sell = abs(delta)

    token_buy_price = round((amount_sell / amount_buy), 9) if amount_buy > 0 and amount_sell > 0 else 0
    return token_buy, amount_buy, token_sell, amount_sell, token_buy_price

def filter_dataframe(df):
    df_success = df[df["TransactionStatus"] == "Success"]
    
    df_filtered = df_success[
        ~(
            (df_success["TokenMintBuy"] == "wSOL") & (df_success["TokenMintSell"].isna())  # wSol
        ) &
        ~(
            (df_success["TokenMintBuy"].isna()) & (df_success["TokenMintSell"].isna())  # Trash transactions
        )
    ]
    
    # Union all with Failed transactions
    df_final = pd.concat([df_filtered, df[df["TransactionStatus"] == "Failed"]], ignore_index=True)
    
    return df_final

def process_transactions(signatures_with_slots, wallet_address):
    data = []
    unique_slots = set(slot for _, slot in signatures_with_slots)

    for slot in unique_slots:
        time.sleep(0.1) 
        block_data = get_block(slot)
        if not block_data:
            continue

        block_time = block_data.get("blockTime", 0)
        block_time_str = unix_to_datetime(block_time) if block_time else "Unknown"
        block_date_str = block_time_str[:10] if block_time else "Unknown"

        filtered_txs = filter_transactions_by_wallet(block_data, wallet_address)
        for tx in filtered_txs:
            tips_amount, tips_destination = extract_tips(tx, wallet_address)
            fail_reason = extract_fail_reason(tx)
            token_buy, amount_buy, token_sell, amount_sell, token_buy_price = extract_trades(tx, wallet_address)

            token_address = token_buy if token_sell == "wSOL" else token_sell
            token_flow = "in" if token_sell == "wSOL" else "out"
            token_real_amount = amount_buy if token_sell == "wSOL" else -amount_sell
            sol_real_amount = -amount_sell if token_sell == "wSOL" else amount_buy
            sol_flow = "out" if token_sell == "wSOL" else "in"

            data.append({
                "Signature": tx["transaction"]["signatures"][0],
                "TransactionStatus": "Success" if tx.get("meta", {}).get("err") is None else "Failed",
                "FailReason": fail_reason,
                "BlockTime": block_time_str,
                "BlockDate": block_date_str,
                "Slot": slot,
                "ComputeUnits": tx.get("meta", {}).get("computeUnitsConsumed", 0),
                "Fee": tx.get("meta", {}).get("fee", 0) / 1e9,
                "TipsAmount": tips_amount,
                "TipsDestination": tips_destination,
                "TokenMintBuy": token_buy,
                "TokenAmountBuy": amount_buy,
                "TokenMintSell": token_sell,
                "TokenAmountSell": amount_sell,
                "TokenBuyPrice": token_buy_price,
                "TokenAddress": token_address,
                "TokenFlow": token_flow,
                "TokenRealAmount": token_real_amount,
                "SOLRealAmount": sol_real_amount,
                "SOLFlow": sol_flow
            })

    df = pd.DataFrame(data)
    return filter_dataframe(df) 

def generate_deals_dataframe(df_transaction_flattened):
    deals = []
    new_deal_list = []

    # All successfull transactions
    success_deals_df = df_transaction_flattened[df_transaction_flattened['TransactionStatus'] == 'Success']
    # All Failed transactions
    fails_deals_df = df_transaction_flattened [df_transaction_flattened['TransactionStatus'] == 'Failed']
    # Sort all transactions
    df_sorted = success_deals_df.sort_values(by=['BlockTime']).reset_index(drop=True)
    grouped_success = df_sorted.groupby('TokenAddress')

    for token, transactions in grouped_success:
        transactions = transactions.reset_index(drop=True)
        current_deal = []  
        total_buy = 0  
        total_sell = 0  

        for idx, row in transactions.iterrows():
            if row['TokenFlow'] == 'in':
                total_buy += row['TokenRealAmount']
                current_deal.append(row)
            elif row['TokenFlow'] == 'out':
                total_sell += abs(row['TokenRealAmount'])
                current_deal.append(row)

            # Check if deal is closed
            if round(total_sell) == round(total_buy) and round(total_buy) > 0:
                deal_df = pd.DataFrame(current_deal)

                # Create final DataFrame
                deals.append({
                    'DealStartTime': deal_df['BlockTime'].min(),
                    'DealEndTime': deal_df['BlockTime'].max(),
                    'DealStartDate': deal_df['BlockDate'].min(),
                    'DealEndDate': deal_df['BlockDate'].max(),
                    'DealStartBlock': deal_df['Slot'].min(),
                    'DealEndBlock': deal_df['Slot'].max(),
                    'TransactionTimeArray': deal_df['BlockTime'].tolist(),
                    'TipsDestinationArray' : deal_df['TipsDestination'].tolist(),
                    'Tips' : deal_df['TipsAmount'].sum(),
                    'Fee' : deal_df['Fee'].sum(),
                    'ComputeUnitsArray' : deal_df['ComputeUnits'].tolist(),
                    'SignatureArray': deal_df['Signature'].tolist(),
                    'TokenAddress': token,
                    'TokenRealAmountArray': deal_df['TokenRealAmount'].tolist(),
                    'SOLRealAmountArray': deal_df['SOLRealAmount'].tolist(),
                    'TokenRealAmountBuy': deal_df.loc[deal_df['TokenFlow'] == 'in', 'TokenRealAmount'].sum(),
                    'TokenRealAmountSell': deal_df.loc[deal_df['TokenFlow'] == 'out', 'TokenRealAmount'].sum(),
                    'SOLRealAmountBuyToken': deal_df.loc[deal_df['SOLFlow'] == 'out', 'SOLRealAmount'].sum(),
                    'SOLRealAmountSellToken': deal_df.loc[deal_df['SOLFlow'] == 'in', 'SOLRealAmount'].sum(),
                    'SOLEarnings': deal_df.loc[deal_df['SOLFlow'] == 'out', 'SOLRealAmount'].sum() +
                                   deal_df.loc[deal_df['SOLFlow'] == 'in', 'SOLRealAmount'].sum() - 
                                   deal_df['TipsAmount'].sum() - deal_df['Fee'].sum(),
                    'SuccessTransactionFlg': 1,
                    'FailReason': deal_df['FailReason'].max(),
                    'DealTimeDiffMinutes' : (pd.to_datetime(deal_df['BlockTime'].max()) - pd.to_datetime(deal_df['BlockTime'].min())).total_seconds() / 60 
                })

                total_buy = 0
                total_sell = 0
                current_deal = []

    
    for _, row in fails_deals_df.iterrows():
        new_deal = {
            'DealStartTime': row['BlockTime'],
            'DealEndTime': None,
            'DealStartDate': row['BlockDate'],
            'DealEndDate': None,
            'DealStartBlock': row['Slot'],
            'DealEndBlock': None,
            'TransactionTimeArray': None,
            'TipsDestinationArray' : None,
            'Tips' : row['TipsAmount'],
            'Fee' : row['Fee'],
            'ComputeUnitsArray' : [row['ComputeUnits']],
            'SignatureArray': [row['Signature']],
            'TokenAddress': token,
            'TokenRealAmountArray': [row['TokenRealAmount']],
            'SOLRealAmountArray': [row['SOLRealAmount']],
            'TokenRealAmountBuy': row['TokenRealAmount'] if row['TokenFlow'] == 'in' else 0,
            'TokenRealAmountSell': row['TokenRealAmount'] if row['TokenFlow'] == 'out' else 0,
            'SOLRealAmountBuyToken': row['SOLRealAmount'] if row['SOLFlow'] == 'out' else 0,
            'SOLRealAmountSellToken': row['SOLRealAmount'] if row['SOLFlow'] == 'in' else 0,
            'SOLEarnings': (row['SOLRealAmount'] if row['SOLFlow'] == 'out' else 0) +
                           (row['SOLRealAmount'] if row['SOLFlow'] == 'in' else 0) - 
                            row['Fee'],
            'SuccessTransactionFlg': 0,
            'FailReason': row['FailReason'],
            'DealTimeDiffMinutes': None
        }
        
        new_deal_list.append(new_deal)
    
    
    failed_deals_df = pd.DataFrame(new_deal_list)
    deals_df = pd.DataFrame(deals)
    
    if not failed_deals_df.empty: 
        deals_df = pd.concat([deals_df, failed_deals_df], ignore_index=True)

    # Add SuccessDealFlg and SOLEarningsPercent
    deals_df['SOLEarningsPercent'] = (deals_df['SOLEarnings'] / deals_df['SOLRealAmountBuyToken'].abs())
    deals_df['SuccessDealFlg'] = deals_df.apply(lambda row: 1 if row['SOLEarnings'] > 0 and row['SuccessTransactionFlg'] == 1 
                                                else (0 if row['SOLEarnings'] <= 0 and row['SuccessTransactionFlg'] == 1 else ""), axis=1)

    deals_df['FailDealFlg'] = deals_df.apply(lambda row: 1 if row['SOLEarnings'] <= 0 and row['SuccessTransactionFlg'] == 1 
                                            else (0 if row['SOLEarnings'] > 0 and row['SuccessTransactionFlg'] == 1 else ""), axis=1)
    deals_df['EarningsSOL'] = deals_df['SOLEarnings'].apply(lambda x: x if x >= 0 else "")
    deals_df['LoosingsSOL'] = deals_df.apply(lambda row: row['SOLEarnings'] if row['SOLEarnings'] < 0 and row['SuccessTransactionFlg'] == 1 else "", axis=1)
    deals_df['DealWeek'] = pd.to_datetime(deals_df['DealStartDate']) - pd.to_timedelta(pd.to_datetime(deals_df['DealStartDate']).dt.weekday, unit='D')
    deals_df['DealMonth'] = pd.to_datetime(deals_df['DealStartDate']).values.astype('datetime64[M]')
    deals_df['FailedTransactionFlg'] = deals_df['SuccessTransactionFlg'].apply(lambda x: 0 if x == 1 else 1)
    
    return deals_df

def append_to_google_sheets(df, spreadsheet_name, sheet_name, service_account_file):
    """
        Adds data from a DataFrame to Google Sheets in the first empty row.
        If the sheet is empty, writes headers and data.
        If the sheet is not empty, add data to the first empty row in column A.

        :param df: pandas DataFrame with data to write
        :param spreadsheet_name: Name of the Google Sheets file
        :param sheet_name: Name of the sheet where to write data
        :param service_account_file: Path to the service account JSON file
    """

    # Authorize in Google Sheets API
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(service_account_file, scope)
    client = gspread.authorize(creds)

    # Opem Google Sheets by GS name
    spreadsheet = client.open(spreadsheet_name)

    try:
        sheet = spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows="1000", cols="20")

    columns_to_export = [
        "DealStartTime", "DealEndTime", "DealStartDate", "DealEndDate", "DealWeek", "DealMonth", "DealStartBlock", "DealEndBlock",
        "Fee", "Tips", "TokenAddress", "TokenRealAmountBuy", "TokenRealAmountSell",
        "SOLRealAmountBuyToken", "SOLRealAmountSellToken", "SOLEarnings", 
        "SOLEarningsPercent", "SuccessTransactionFlg", "FailedTransactionFlg", "SuccessDealFlg", "FailDealFlg", "EarningsSOL", "LoosingsSOL",
        "DealTimeDiffMinutes"
    ]
    
    # Clean our df
    deals_subset = df[columns_to_export] if set(columns_to_export).issubset(df.columns) else df
    deals_subset = deals_subset.replace([np.inf, -np.inf], np.nan).fillna("")

    # Checking if there is already data in Google Sheets
    existing_data = sheet.col_values(1)  

    if len(existing_data) == 0:  # If sheet is empty then start from 1 row
        start_row = 1
        include_headers = True
    else:  # The sheet is not empty, find the first free row in column A
        start_row = len(existing_data) + 1
        include_headers = False

    # Add data to the sheet
    set_with_dataframe(sheet, deals_subset, row=start_row, include_column_header=include_headers)

    print(f"✅ Data successfully added to '{sheet_name}' starting at row {start_row}!")

# 📌 Send message function in Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("✅ Message successfully sent to Telegram!")
    else:
        print(f"❌ Error sending message: {response.text}")

def send_daily_summary():
    # Authorization in Google Sheets API
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    client = gspread.authorize(creds)

    # Reading data from Google Sheets
    sheet = client.open(SPREADSHEET_NAME).worksheet(SHEET_NAME)
    data = sheet.get_all_records()

    if not data:
        print("❌ No data in Google Sheets!")
        return

    df = pd.DataFrame(data)
    df["DealStartDate"] = pd.to_datetime(df["DealStartDate"])

    # Let's take the last 2 days
    df_sorted = df.sort_values(by="DealStartDate", ascending=False)
    last_two_days = df_sorted["DealStartDate"].unique()[:2]

    df_latest = df_sorted[df_sorted["DealStartDate"] == last_two_days[0]]
    df_previous = df_sorted[df_sorted["DealStartDate"] == last_two_days[1]] if len(last_two_days) > 1 else None

    def calculate_metrics(df):
        total_earnings = df["SOLEarnings"].sum()
        total_transactions = len(df)
        total_deals = df["SuccessTransactionFlg"].sum()
        tatal_deals_without_slip = len(df[df["SuccessTransactionFlg"] == 1])
        success_deals = len(df[df["SOLEarnings"] > 0])
        slippage_deals = len(df[df["FailedTransactionFlg"] == 1])

        success_rate = (success_deals / tatal_deals_without_slip * 100) if tatal_deals_without_slip > 0 else 0
        slippage_rate = (slippage_deals / total_transactions * 100) if total_transactions > 0 else 0
        avg_profit = df[df["SOLEarnings"] > 0]["SOLEarnings"].mean() if success_deals > 0 else 0
        avg_loss = df[df["LoosingsSOL"] != ""]["LoosingsSOL"].mean() if not df[df["LoosingsSOL"] != ""].empty else 0

        return {
            "total_earnings": total_earnings,
            "total_transactions": total_transactions,
            "total_deals": total_deals,
            "success_rate": success_rate,
            "slippage_rate": slippage_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss
        }

    # Calculate metrics
    latest_metrics = calculate_metrics(df_latest)
    previous_metrics = calculate_metrics(df_previous) if df_previous is not None else None

    def compare_metrics(new, old):
        if old is None:
            return ""  # If there is no data for the previous day, do not add a comparison
        if old == 0:
            return ""  # If it was 0 before, we don't count the percentage
        change = ((new - old) / abs(old)) * 100
        arrow = "⬆️" if change > 0 else "⬇️"
        return f"({change:+.1f}% {arrow})"

    summary_message = f"""
📊 *Daily report for {last_two_days[0].strftime('%Y-%m-%d')}* 📊

    💰 *Net Income*: `{latest_metrics['total_earnings']:.4f} SOL` {compare_metrics(latest_metrics['total_earnings'], previous_metrics['total_earnings'])}
    📈 *Transaction Count*: `{latest_metrics['total_transactions']}` {compare_metrics(latest_metrics['total_transactions'], previous_metrics['total_transactions'])}
    ⚠️ *% Slippage*: `{latest_metrics['slippage_rate']:.1f}%` {compare_metrics(latest_metrics['slippage_rate'], previous_metrics['slippage_rate'])}
    💸 *Deals Count*: `{latest_metrics['total_deals']}` {compare_metrics(latest_metrics['total_deals'], previous_metrics['total_deals'])}
    ✅ *% Success Rate*: `{latest_metrics['success_rate']:.1f}%` {compare_metrics(latest_metrics['success_rate'], previous_metrics['success_rate'])}
    📊 *Average Income*: `{latest_metrics['avg_profit']:.4f} SOL` {compare_metrics(latest_metrics['avg_profit'], previous_metrics['avg_profit'])}
    📉 *Average Loss*: `{latest_metrics['avg_loss']:.4f} SOL` {compare_metrics(latest_metrics['avg_loss'], previous_metrics['avg_loss'])}

[Dashboard Link]({SPREADSHEET_LINK})
    """.strip()

    send_telegram_message(summary_message)
    
def send_weekly_summary():

    # Authorization in Google Sheets API
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    client = gspread.authorize(creds)

    # Reading data from Google Sheets
    sheet = client.open(SPREADSHEET_NAME).worksheet(SHEET_NAME)
    data = sheet.get_all_records()

    if not data:
        print("❌ No data in Google Sheets!")
        return

    df = pd.DataFrame(data)
    df["DealStartDate"] = pd.to_datetime(df["DealStartDate"])

    # Determine dates for comparison
    today = datetime.now().date() - timedelta(days=datetime.now().weekday())
    last_monday = today - timedelta(days=today.weekday() + 7)
    previous_monday = last_monday - timedelta(days=7) 

    df_last_week = df[(df["DealStartDate"] >= pd.Timestamp(last_monday)) & (df["DealStartDate"] < pd.Timestamp(last_monday + timedelta(days=7)))]
    df_prev_week = df[(df["DealStartDate"] >= pd.Timestamp(previous_monday)) & (df["DealStartDate"] < pd.Timestamp(last_monday))]

    def calculate_metrics(df):
        total_earnings = df["SOLEarnings"].sum()
        total_transactions = len(df)
        total_deals = df["SuccessTransactionFlg"].sum()
        tatal_deals_without_slip = len(df[df["SuccessTransactionFlg"] == 1])
        success_deals = len(df[df["SOLEarnings"] > 0])
        slippage_deals = len(df[df["FailedTransactionFlg"] == 1])

        success_rate = (success_deals / tatal_deals_without_slip * 100) if tatal_deals_without_slip > 0 else 0
        slippage_rate = (slippage_deals / total_transactions * 100) if total_transactions > 0 else 0
        avg_profit = df[df["SOLEarnings"] > 0]["SOLEarnings"].mean() if success_deals > 0 else 0
        avg_loss = df[df["LoosingsSOL"] != ""]["LoosingsSOL"].mean() if not df[df["LoosingsSOL"] != ""].empty else 0

        return {
            "total_earnings": total_earnings,
            "total_transactions": total_transactions,
            "total_deals": total_deals,
            "success_rate": success_rate,
            "slippage_rate": slippage_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss
        }

    # Calculating metrics
    last_week_metrics = calculate_metrics(df_last_week)
    prev_week_metrics = calculate_metrics(df_prev_week) if not df_prev_week.empty else None

    def compare_metrics(new, old):
        if old is None:
            return ""  # If there is no data for the previous week, do not add a comparison
        if old == 0:
            return ""  # If it was 0 before, we don't count the percentage
        change = ((new - old) / abs(old)) * 100
        arrow = "⬆️" if change > 0 else "⬇️"
        return f"({change:+.1f}% {arrow})"

    summary_message = f"""
📊 *Weekly report for the period {last_monday.strftime('%Y-%m-%d')} - {(last_monday + timedelta(days=6)).strftime('%Y-%m-%d')}* 📊

    💰 *Net Income*: `{last_week_metrics['total_earnings']:.4f} SOL` {compare_metrics(last_week_metrics['total_earnings'], prev_week_metrics['total_earnings'])}
    📈 *Transaction Count*: `{last_week_metrics['total_transactions']}` {compare_metrics(last_week_metrics['total_transactions'], prev_week_metrics['total_transactions'])}
    ⚠️ *% Slippage*: `{last_week_metrics['slippage_rate']:.1f}%` {compare_metrics(last_week_metrics['slippage_rate'], prev_week_metrics['slippage_rate'])}
    💸 *Deals Count*: `{last_week_metrics['total_deals']}` {compare_metrics(last_week_metrics['total_deals'], prev_week_metrics['total_deals'])}
    ✅ *% Success Rate*: `{last_week_metrics['success_rate']:.1f}%` {compare_metrics(last_week_metrics['success_rate'], prev_week_metrics['success_rate'])}
    📊 *Average Income*: `{last_week_metrics['avg_profit']:.4f} SOL` {compare_metrics(last_week_metrics['avg_profit'], prev_week_metrics['avg_profit'])}
    📉 *Average Loss*: `{last_week_metrics['avg_loss']:.4f} SOL` {compare_metrics(last_week_metrics['avg_loss'], prev_week_metrics['avg_loss'])}

[Dashboard Link]({SPREADSHEET_LINK})
    """.strip()

    send_telegram_message(summary_message)
    
# Function for collecting data and recording in Google Sheets
def fetch_and_store_data():
    print(f"Starting screept: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # 📅 Определяем вчерашний день
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1) 
    start_date = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc)
    end_date = datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59, tzinfo=timezone.utc)

    signatures = get_signatures_for_wallet(WALLET_ADDRESS, start_date, end_date)
    df = process_transactions(signatures, WALLET_ADDRESS)
    deals_df = generate_deals_dataframe(df).sort_values(by='DealStartTime', ascending=True).reset_index(drop=True)

    append_to_google_sheets(deals_df, SPREADSHEET_NAME, SHEET_NAME, SERVICE_ACCOUNT_FILE)
    print(f"✅ Data for {yesterday.strftime('%Y-%m-%d')} successfully written to Google Sheets!")
    
    send_daily_summary()
    print(f"✅ Statistics for {yesterday.strftime('%Y-%m-%d')} successfully sent to TG!")

# 📌 Shedule task
schedule.every().day.at("07:30").do(fetch_and_store_data)

# 📌 Add monday shedule for weekly summary
schedule.every().monday.at("07:30").do(send_weekly_summary)

print("⏳ Scheduler started. Expecting 07:30 UTC every day...")

while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
