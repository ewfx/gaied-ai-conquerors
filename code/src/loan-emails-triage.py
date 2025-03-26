import os
import email
import json
import pandas as pd
import PyPDF2
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Paths
EMAILS_FOLDER = "banking-loan-emails-attachments-data/"
CONFIG_FILE = "config/banking_config.json"
openai.api_key = "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"

# Load configuration
with open(CONFIG_FILE, "r") as f:
    CONFIG = json.load(f)

# Initialize LangChain LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
classification_prompt = PromptTemplate(
    input_variables=["email_content", "rules"],
    template=(
        "You are an AI trained to classify loan processing requests based on the following rules: {rules}.\n\n"
        "Given the email content below, identify the request type and sub-request type.\n\nEmail Content:\n{email_content}\n\n"
        "Request Type:\nSub-Request Type:"
    )
)
classifier_chain = LLMChain(llm=llm, prompt=classification_prompt)

# Function to process email files
def process_email(filepath):
    with open(filepath, "rb") as f:
        msg = email.message_from_binary_file(f)

    subject = msg["subject"]
    sender = msg["from"]

    # Extract email body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body = part.get_payload(decode=True).decode()
    else:
        body = msg.get_payload(decode=True).decode()

    # Extract attachments
    attachments = []
    for part in msg.walk():
        if part.get_content_disposition() == "attachment":
            filename = part.get_filename()
            if filename:
                attachments.append({"filename": filename, "content": part.get_payload(decode=True)})

    return {
        "subject": subject,
        "sender": sender,
        "body": body,
        "attachments": attachments
    }

# Function to classify emails using LLM
def classify_email(email_body):
    rules = f"Request Types: {CONFIG['request_types']}. Sub-Request Types: {CONFIG['sub_request_types']}."
    classification_result = classifier_chain.run(email_content=email_body, rules=rules)
    request_type, sub_request_type = classification_result.split("\n")
    return request_type.strip(), sub_request_type.strip()

# Function to parse PDF attachments
def parse_pdf(content):
    try:
        pdf_reader = PyPDF2.PdfReader(content)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# Function to parse Excel attachments
def parse_excel(content):
    try:
        # Save the content temporarily
        with open("temp.xlsx", "wb") as f:
            f.write(content)
        workbook = openpyxl.load_workbook("temp.xlsx")
        sheet = workbook.active
        data = []
        for row in sheet.iter_rows(values_only=True):
            data.append(row)
        os.remove("temp.xlsx")  # Clean up temp file
        return data
    except Exception as e:
        return f"Error reading Excel: {e}"

# Parse attachments
def parse_attachments(attachments):
    parsed_data = []
    for attachment in attachments:
        filename = attachment["filename"]
        content = attachment["content"]
        if filename.endswith(".pdf"):
            parsed_data.append({"filename": filename, "content": parse_pdf(content)})
        elif filename.endswith(".xlsx"):
            parsed_data.append({"filename": filename, "content": parse_excel(content)})
        else:
            parsed_data.append({"filename": filename, "content": "Unsupported file type"})
    return parsed_data

# Detect duplicates using TF-IDF
def detect_duplicates(emails_df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(emails_df["body"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    duplicates = []
    for i in range(len(cosine_sim)):
        if any(cosine_sim[i][j] > 0.8 and i != j for j in range(len(cosine_sim))):
            duplicates.append(True)
        else:
            duplicates.append(False)
    emails_df["is_duplicate"] = duplicates
    return emails_df

# Process all emails in the folder
def process_emails_from_folder(folder_path):
    all_emails = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".eml"):
            email_path = os.path.join(folder_path, filename)
            email_data = process_email(email_path)
            all_emails.append(email_data)
    return all_emails

# Main function
def main():
    # Step 1: Process emails
    emails = process_emails_from_folder(EMAILS_FOLDER)
    emails_df = pd.DataFrame(emails)

    # Step 2: Parse attachments
    emails_df["parsed_attachments"] = emails_df["attachments"].apply(parse_attachments)

    # Step 3: Classify emails
    emails_df[["request_type", "sub_request_type"]] = emails_df["body"].apply(
        lambda body: pd.Series(classify_email(body))
    )

    # Step 4: Detect duplicates
    emails_df = detect_duplicates(emails_df)

    # Step 5: Sort by priority
    emails_df["priority"] = emails_df["request_type"].map(CONFIG["priorities"])
    emails_df = emails_df.sort_values(by="priority")

    # Display results
    print(emails_df)
    return emails_df

# Run the main function
if __name__ == "__main__":
    classified_emails = main()
