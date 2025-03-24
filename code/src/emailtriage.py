import os
import configparser
import openai

# Set up your OpenAI API key
openai.api_key = "your-openai-api-key"

def load_properties(file_path):
    """Loads keys, rules, and priorities from a .properties file."""
    config = configparser.ConfigParser()
    try:
        config.read(file_path)
        return config
    except Exception as e:
        print(f"Error loading properties file: {e}")
        return None

def analyze_attachment(file_path, config):
    """Analyzes the content of an attachment using Generative AI."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract analysis parameters from the properties file
        rules = config['RULES']
        keys = config['KEYS']
        priorities = config['PRIORITIES']

        # Prompt for generative AI analysis
        prompt = f"""
        Analyze the following content according to these rules, keys, and priorities:
        Rules: {rules}
        Keys: {keys}
        Priorities: {priorities}

        Content: {content}
        """

        # Using the new OpenAI Chat API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with desired model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        return f"Error analyzing file {file_path}: {e}"

def process_attachments(folder_path, config):
    """Processes all attachments in the specified folder."""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # Check if it's a file
            print(f"Analyzing {file_name}...")
            result = analyze_attachment(file_path, config)
            print(result)

if __name__ == "__main__":
    # Path to your properties file and attachments folder
    properties_file = "emailtriage-config.properties"  # Path to your .properties file
    attachments_folder = "emails_with-attachments-data"  # Folder with email attachments

    # Load properties
    config = load_properties(properties_file)
    if config is not None:
        # Process email attachments
        process_attachments(attachments_folder, config)
    else:
        print("Could not load properties. Exiting.")

