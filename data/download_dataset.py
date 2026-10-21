"""
Dataset Downloader for SMS Spam Collection Dataset.
Downloads the UCI SMS Spam Collection dataset from UCI ML Repository.
"""

import urllib.request
import zipfile
import os
import pandas as pd


def download_sms_spam_dataset(save_dir: str = "data") -> str:
    """
    Download the SMS Spam Collection dataset from UCI ML Repository.

    Parameters
    ----------
    save_dir : str
        Directory to save the dataset.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "sms_spam.csv")

    if os.path.exists(csv_path):
        print(f"[INFO] Dataset already exists at: {csv_path}")
        return csv_path

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = os.path.join(save_dir, "smsspamcollection.zip")

    print("[INFO] Downloading SMS Spam Collection dataset...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("[INFO] Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(save_dir)

        raw_path = os.path.join(save_dir, "SMSSpamCollection")
        df = pd.read_csv(raw_path, sep="\t", header=None, names=["label", "message"])
        df.to_csv(csv_path, index=False)
        os.remove(zip_path)
        os.remove(raw_path)
        print(f"[INFO] Dataset saved to: {csv_path}")
        return csv_path

    except Exception as e:
        print(f"[WARN] Download failed ({e}). Generating synthetic fallback dataset...")
        return _generate_fallback_dataset(csv_path)


def _generate_fallback_dataset(csv_path: str) -> str:
    """
    Generate a small synthetic SMS spam/ham dataset as fallback.

    Parameters
    ----------
    csv_path : str
        Path to save the generated CSV.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    spam_messages = [
        "WINNER!! You have been selected to receive a $1000 gift card. Call now!",
        "FREE entry in 2 a weekly competition to win FA Cup final tickets! Text FA to 87121",
        "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
        "URGENT: Your account will be suspended. Verify now at http://fakebank.com",
        "You have won 1 million dollars. Send your bank details to claim your reward.",
        "FREE msg: 3 months free broadband. Apply NOW txt BROADBAND to 88877",
        "Claim your FREE gift now! Reply YES to receive a special voucher worth $500",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 to 87575",
        "PRIVATE! Your 2004 account statement for shows 800 un-redeemed S. I. M. points",
        "Had your mobile 11 months or more? You are entitled to update to the latest colour mobile",
        "Want explicit CHAT? Call 08000930705 now! 150p per min. 18+",
        "IMPORTANT - You could be entitled up to 3160 in compensation from mis-sold PPI on a credit card.",
        "Please call our customer service representative on FREEPHONE 0800 0431070 between 9am-11pm",
        "Todays Voda numbers ending 7548 are selected to receive a $350 award. Call 087187272008",
        "You are a WINNER! Your mobile number has been randomly selected for a cash prize!",
        "Double your money TODAY! Secret investment strategy revealed. Act NOW!",
        "EARN UP TO $500 PER DAY WORKING FROM HOME! No experience needed. Click here.",
        "BANK ALERT: Suspicious activity detected. Verify your account immediately.",
        "SPECIAL OFFER: Buy one get one FREE on all products. Limited time only!",
        "You have a pending payment. Confirm your details at: www.secure-payment.net",
        "Win a luxury cruise for 2! Enter our sweepstakes now. Text CRUISE to 55555.",
        "Your loan has been pre-approved! Get up to $50,000. Apply in minutes.",
        "ALERT: Your password has been compromised. Reset immediately at securelogin.net",
        "Congratulations! As a valued customer, you've been selected for a special deal.",
        "FREE RINGTONES! Text RING to 84484. 3 ringtones/week, $4.50/wk until cancel.",
        "Stop wasting money! Get cheap car insurance from just $99/month. Call now!",
        "You have 1 unread notification from your bank. Login to check: onlinebanking.xyz",
        "JACKPOT WINNER! Claim your $2,500,000 prize today. Contact prize@lottery.com",
        "Make money fast with our proven system. Over 10,000 satisfied customers!",
        "Your credit score can be repaired in 30 days! Call 1-800-FIX-CREDIT now.",
        "HOT SINGLES in your area are waiting to chat! Call 0906 363 7382 now!",
        "ROFL! Your mobile has been selected to win a prize. Collect at www.prize4u.co.uk",
        "Save money on your phone bill! Switch to our network and get 50% off for 6 months.",
        "REMINDER: Your free gift is waiting. Don't miss out! Expires midnight tonight.",
        "You've been selected as one of 10 winners of the national lottery draw!",
        "HELLO! You have an important package waiting. Confirm delivery at: ups-track.xyz",
        "Earn passive income from home: $1000/day guaranteed. No skills required!",
        "Your subscription has expired. Renew now to avoid losing your account permanently.",
        "We tried to deliver your parcel. Schedule redelivery: parceltrack.free-site.net",
        "Text YES to 55555 to claim your free vacation package to Cancun!",
    ]

    ham_messages = [
        "Hey, are you coming to the meeting tomorrow at 3pm?",
        "Can you pick up some milk on your way home?",
        "Happy birthday! Hope you have a wonderful day.",
        "I'll be there in about 20 minutes. Traffic is heavy.",
        "Don't forget about our dinner reservation tonight at 7.",
        "The project deadline has been moved to next Friday.",
        "Thanks for the help yesterday, really appreciated it!",
        "Are you free this weekend? Let's catch up.",
        "I just finished the report, sending it over now.",
        "Your package has been shipped and will arrive by Thursday.",
        "Mom says hi and wants to know if you're coming for Christmas.",
        "Call me when you get a chance, need to talk to you.",
        "The kids are asking about the trip. Have you booked the hotel yet?",
        "Reminder: Doctor appointment tomorrow at 10:30 AM.",
        "Just saw your email, will respond in detail tomorrow.",
        "Running a bit late. Please start without me.",
        "Can we reschedule our call to Wednesday instead?",
        "The weather looks nice this weekend! Want to go hiking?",
        "I found the keys! They were under the couch the whole time.",
        "Good morning! Don't forget to bring the presentation files.",
        "Your order has been confirmed. Expected delivery: 3-5 business days.",
        "Hey, did you watch the game last night? What a match!",
        "I'm at the grocery store. Do we need anything else?",
        "The meeting was moved from 2pm to 4pm today.",
        "Please review the attached document and let me know your thoughts.",
        "Thinking of you! Hope everything's going well.",
        "Got the report. Looks great, just a few minor edits needed.",
        "Can you send me the address for the party on Saturday?",
        "Flight confirmed for next Tuesday. Boarding at Gate 12.",
        "Study group at the library at 5pm. Don't forget your notes!",
        "The car service appointment is confirmed for 9am tomorrow.",
        "Let me know what time works for you to grab coffee.",
        "I need to talk to you about something important. Call me when free.",
        "Nah I don't think he goes to usf, he lives around here though.",
        "Even my brother is not like to speak with me. They treat me like aids patent.",
        "I HAVE A DATE ON SUNDAY WITH WILL!",
        "As per your request the song has been set as your callertune for 30 days",
        "Ok lar... Joking wif u oni... Ok or not? Happy anot?",
        "Fine if that's the way u feel. That's the way its gota b",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?",
        "I've been searching for the right words to thank you for this breather.",
        "Did you catch the news this morning? Crazy stuff happening.",
        "Let's plan a road trip next month. You in?",
        "The dinner was amazing! That restaurant is a keeper.",
        "Just a reminder that the library books are due back tomorrow.",
        "Your blood test results are back. Please call to discuss.",
        "We're having a team lunch on Thursday. Hope you can join!",
        "I passed my exam! Thanks for all your help and support.",
        "The wifi is down again. Can you check the router?",
        "Can you email me the files before the meeting?",
        "It was so nice seeing you at the reunion! Let's stay in touch.",
        "The gym classes are full for Tuesday. Want to try Wednesday?",
        "She's alright now, just had a minor cold.",
        "I'm in the area this weekend, want to grab lunch?",
        "Please confirm that you received my previous email.",
        "Just checking if you're okay. Haven't heard from you in a while.",
        "The conference has been rescheduled for November 15th.",
        "Happy anniversary! Can't believe it's been 5 years already.",
        "I'll drop off the documents at your office by noon.",
        "The kids had a great time at soccer practice today!",
    ]

    messages = spam_messages + ham_messages
    labels = ["spam"] * len(spam_messages) + ["ham"] * len(ham_messages)

    df = pd.DataFrame({"label": labels, "message": messages})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Synthetic fallback dataset saved to: {csv_path} ({len(df)} records)")
    return csv_path


if __name__ == "__main__":
    path = download_sms_spam_dataset()
    df = pd.read_csv(path)
    print(f"\nDataset shape: {df.shape}")
    print(df["label"].value_counts())
