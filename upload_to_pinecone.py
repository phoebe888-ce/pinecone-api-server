import csv
import chardet
from pinecone_utils import upload_to_pinecone

def detect_encoding(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read(10000)
    result = chardet.detect(raw)
    return result['encoding']

def load_csv_data(filepath: str):
    data = []
    encoding = detect_encoding(filepath)
    print(f"Detected file encoding: {encoding}")

    with open(filepath, newline='', encoding=encoding) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = f"Customer: {row['Email Summary']}\nAI: {row['Ideal Reply']}"
            metadata = {
                "threadId": row["Email ID"],
                "customerMsg": row["Email Summary"],
                "aiReply": row["Ideal Reply"],
                "timestamp": ""
            }
            data.append({"id": row["Email ID"], "text": text, "metadata": metadata})
    return data

if __name__ == "__main__":
    dataset = load_csv_data("emails.csv")
    print(f"Uploading {len(dataset)} emails to Pinecone...")
    upload_to_pinecone(dataset)
    print("✅ Upload complete.")