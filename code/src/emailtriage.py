import os
import configparser
import openai  # Requires OpenAI library for GenAI

# Load your OpenAI API key
openai.api_key = "your-openai-api-key"

def load_properties(file_path):
    """Loads keys, rules, and priorities from a .properties file."""
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def analyze_attachment(file_path, config):
    """Analyzes the content of an attachment using GenAI."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract analysis parameters
        rules = config['RULES']
        keys = config['KEYS']
        priorities = config['PRIORITIES']
        print(priorities)
        
        # Sample prompt for generative AI
        prompt = f"""
        Analyze the following content according to these rules, keys, and priorities:
        Rules: {rules}
        Keys: {keys}
        Priorities: {priorities}
        
        Content: {content}
        """
        # Call the GenAI model
        response = openai.Completion.create(
            engine="text-davinci-003",  # Example engine
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

    except Exception as e:
        return f"Error analyzing file {file_path}: {e}"

def process_attachments(folder_path, config):
    """Processes all attachments in the given folder."""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # Ensure it's a file
            print(f"Analyzing {file_name}...")
            result = analyze_attachment(file_path, config)
            print(result)

if __name__ == "__main__":
    properties_file = "emailtriage-config.properties"  # Path to your .properties file
    attachments_folder = "emails_with-attachments-data"  # Path to the folder with attachments
    
    # Load properties
    config = load_properties(properties_file)
    
    # Process email attachments
    process_attachments(attachments_folder, config)
