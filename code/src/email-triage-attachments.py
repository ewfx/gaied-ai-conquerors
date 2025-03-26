import os
import configparser
import openai
import email


# Set your OpenAI API key
openai.api_key = "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"

# Function to load the configuration from a .properties file
def load_properties(file_path):
    """Loads keys, rules, and priorities from a .properties file."""
    config = configparser.ConfigParser()
    try:
        config.read(file_path)
        return config
    except Exception as e:
        print(f"Error loading properties file: {e}")
        return None

# Function to analyze content using OpenAI
def analyze_content(content, config):
    """Analyzes content using Generative AI."""
    try:
        # Extract parameters from the properties file
        rules = config['RULES']
        keys = config['KEYS']
        priorities = config['PRIORITIES']

        # Create a prompt for generative AI analysis
        prompt = f"""
        Analyze the following content according to these rules, keys, and priorities:
        Rules: {rules}
        Keys: {keys}
        Priorities: {priorities}

        Content: {content}
        """

        # Call OpenAI ChatCompletion API
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Replace with your desired model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        return f"Error analyzing content: {e}"

# Function to process emails in a folder
def process_emails(folder_path, config):
    """Processes .eml files in a specified folder."""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.eml') and os.path.isfile(file_path):  # Process .eml files only
            print(f"Processing email: {file_name}")
            try:
                # Read the email file
                with open(file_path, 'rb') as eml_file:
                    msg = email.message_from_binary_file(eml_file)

                # Extract email content for analysis
                email_body = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        email_body += part.get_payload(decode=True).decode('utf-8', errors='ignore')

                # Analyze email body content
                if email_body:
                    print(f"Analyzing email body...")
                    result = analyze_content(email_body, config)
                    print(result)

                # Process attachments within the email
                for part in msg.walk():
                    if part.get_content_disposition() == "attachment":
                        attachment_name = part.get_filename()
                        if attachment_name:
                            print(f"Analyzing attachment: {attachment_name}")
                            attachment_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            attachment_result = analyze_content(attachment_content, config)
                            print(attachment_result)

            except Exception as e:
                print(f"Error processing email {file_name}: {e}")

# Main function
if __name__ == "__main__":
    # File paths and configuration
    properties_file = "banking-email-triage-config.properties"  # Path to your .properties file
    emails_with_attachments_folder = "banking-emails-attachments-data"  # Folder with email attachments

    # Step 1: Load properties configuration
    config = load_properties(properties_file)
    if config is not None:
        # Step 2: Process emails and analyze their content and attachments
        process_emails(emails_with_attachments_folder, config)
    else:
        print("Could not load properties. Exiting.")


def classify_sub_request(request_type, email_text):
    if request_type == "Closing Notice":
        if "reallocation fee" in email_text.lower():
            return "Reallocation Fee"
        elif "amendment fee" in email_text.lower():
            return "Amendment Fee"
    elif request_type == "Money Movement Inbound":
        if "principal" in email_text.lower():
            return "Principal"
        elif "interest" in email_text.lower():
            return "Interest"
    # Add more rules/subtypes as needed
    return "Unknown"

emails_df['sub_request_type'] = emails_df.apply(
    lambda row: classify_sub_request(row['request_type'], row['body']), axis=1
)



from transformers import pipeline

# Load a pre-trained Hugging Face model for classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def classify_request(email_text):
    result = classifier(email_text)
    return result[0]['label']

emails_df['request_type'] = emails_df['body'].apply(classify_request)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(emails_df['body'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Flag duplicates based on similarity threshold
emails_df['is_duplicate'] = [any(row > 0.8) for row in cosine_sim]
