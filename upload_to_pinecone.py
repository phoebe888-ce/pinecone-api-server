import csv
import chardet
from pinecone_utils import upload_to_pinecone
def detect_encoding(filepath):
    """自动检测文件编码"""
    with open(filepath, 'rb') as f:
        raw = f.read(10000)
    result = chardet.detect(raw)
    return result['encoding']

def load_csv_data(filepath: str):
    """
    加载 CSV 数据，字段包括：
    Email ID, Sender, Email Summary, Issue Type, Ideal Reply
    """
    data = []
    encoding = detect_encoding(filepath)
    print(f"Detected file encoding: {encoding}")

    with open(filepath, newline='', encoding=encoding) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = f"{row['Email Summary']} (Issue Type: {row['Issue Type']})"
            metadata = {
                "email_id": row["Email ID"],
                "sender": row["Sender"],
                "ideal_reply": row["Ideal Reply"],
                "issue_type": row["Issue Type"],
                "email_summary": row["Email Summary"]
            }
            data.append({"id": row["Email ID"], "text": text, "metadata": metadata})
    return data


if __name__ == "__main__":
    dataset = load_csv_data("emails.csv")
    print(f"Uploading {len(dataset)} emails to Pinecone...")
    upload_to_pinecone(dataset)
    print("✅ Upload complete.")
