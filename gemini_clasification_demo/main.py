"""
Gemini Spots Classifier
A script that uses Google's Gemini AI to classify advertising spots based on their descriptions and brand information.
"""

import os
import re
from typing import Tuple, List
import google.generativeai as genai
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def load_and_prepare_data(file_path: str, subset_size: int = 222) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare the dataset for classification.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset
        subset_size (int): Number of rows to use from the dataset
        
    Returns:
        Tuple containing:
        - Training dataset
        - Test dataset
        - Original test dataset with true labels
    """
    df = pd.read_csv(file_path)
    df_subset = df.iloc[:subset_size]
    
    # Split the data
    dataset_train, dataset_test = train_test_split(df_subset, test_size=0.2, random_state=42)
    
    # Clean and prepare datasets
    dataset_train = dataset_train.rename(columns={"Media URL/canal": "MediaURL/canal"})
    dataset_test = dataset_test.rename(columns={"Media URL/canal": "MediaURL/canal"})
    
    # Select relevant columns
    dataset_train = dataset_train[dataset_train.columns[:4]]
    dataset_test_original = dataset_test[dataset_test.columns[:4]]
    
    # Rename columns for clarity
    dataset_test_original = dataset_test_original.rename(columns={"Vertical": "OriginalVertical"})
    dataset_train = dataset_train.rename(columns={"Vertical": "OriginalVertical"})
    
    # Prepare test dataset
    columns_to_keep = ['Description', 'MediaURL/canal', 'Marca']
    dataset_test = dataset_test[columns_to_keep]
    
    return dataset_train, dataset_test, dataset_test_original

def create_gemini_model() -> genai.GenerativeModel:
    """
    Create and configure the Gemini model.
    
    Returns:
        Configured Gemini model instance
    """
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

def extract_predictions_from_response(response_text: str) -> pd.DataFrame:
    """
    Extract predictions from Gemini's response and create a DataFrame.
    
    Args:
        response_text (str): Raw response text from Gemini
        
    Returns:
        DataFrame containing the predictions
    """
    pattern = r"\| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \|"
    matches = re.findall(pattern, response_text)
    
    predictions_df = pd.DataFrame(matches, columns=['Description', 'MediaURL/canal', 'Marca', 'PredictedVertical', 'Explanation'])
    predictions_df = predictions_df.iloc[1:]  # Drop header row
    predictions_df = predictions_df.reset_index(drop=True)
    predictions_df = predictions_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    return predictions_df

def main():
    """Main execution function"""
    try:
        # Load and prepare data
        dataset_train, dataset_test, dataset_test_original = load_and_prepare_data("DummySpots.csv")
        
        # Create Gemini model
        model = create_gemini_model()
        
        # Start chat session
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": ["Analyze the following dataset:"]},
                {"role": "user", "parts": [dataset_train.to_string()]},
                {"role": "user", "parts": ["Based on this analysis, predict the Vertical for the following dataset. This dataset contains multiple ads that are very similar in terms of their 'Description' and 'Marca'. Please ensure that your predictions for similar ads are consistent. Adding an explanation for each row."]},
                {"role": "user", "parts": [dataset_test.to_string()]},
            ]
        )
        
        # Get predictions
        response = chat_session.send_message(
            "The Vertical column represents the category of the advertisement, which is based on the content of the ad. Please predict the Vertical for the test set. If you encounter a situation where you are unsure how to categorize the ad, please predict the Vertical as 'Others.' Include a brief explanation for your prediction in the 'Explanation' column. Please format your response as a table like this:\n\n"
            "| Description | MediaURL/canal | Marca | PredictedVertical | Explanation |\n|---|---|---|---|---|"
        )
        
        # Process predictions
        predictions_df = extract_predictions_from_response(response.text)
        predictions_df.to_csv('results_0.csv', index=False)
        
        # Print statistics
        print(f"Total dataset_train: {len(dataset_train)}")
        print(f"Total dataset_test: {len(dataset_test)}")
        print(f"Total dataset_test_original: {len(dataset_test_original)}")
        print(f"Total predictions_df: {len(predictions_df)}")
        
        # Process missing categorizations
        df_missing_cathegorize = pd.merge(
            dataset_test_original, 
            predictions_df, 
            on=['Description', 'MediaURL/canal', 'Marca'], 
            how='left', 
            indicator=True
        )
        
        df_missing_cathegorize = df_missing_cathegorize.drop(columns=['OriginalVertical'])
        df_missing_cathegorize.to_csv('results_1.csv', index=False)
        
        # Get additional predictions for missing categories
        response = chat_session.send_message(
            f"Please complete the 'PredictedVertical' column in the following dataset."
            f" The dataset contains rows where '_merge' column is 'left_only' and 'PredictedVertical' column is empty."  
            f" So, for those rows, please use the 'Description' and 'Marca' columns to find the closest matching rows in the same dataset which already has a PredictedVertical filled."
            f" If there is no match or you are unsure, predict 'Others' and provide an explanation."
            f" Please format your response as a table like this:\n\n"
            f"| Description | MediaURL/canal | Marca | PredictedVertical | Explanation | _merge |\n|---|---|---|---|---|---|"
        )
        
        # Process additional predictions
        table_pattern = r"\| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \|"
        second_matches = re.findall(table_pattern, response.text)
        
        second_matches = pd.DataFrame(second_matches, columns=['Description', 'MediaURL/canal', 'Marca', 'PredictedVertical', 'Explanation', '_merge'])
        second_matches = second_matches.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        second_matches = second_matches.iloc[1:]
        second_matches = second_matches.reset_index(drop=True)
        
        second_matches.to_csv('results_1_1.csv', index=False)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 