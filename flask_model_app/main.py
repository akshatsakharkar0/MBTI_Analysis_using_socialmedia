# Step 1: Load into Pandas
import pandas as pd

df = pd.read_csv('mbti_extended_questions_dataset.csv')
print("✅ Loaded", len(df), "questions")
df.head(5)  # Preview first 5
# Step 3: MBTI Quiz Engine
def run_mbti_quiz(df):
    scores = {"E": 0, "I": 0, "S": 0, "N": 0, "T": 0, "F": 0, "J": 0, "P": 0}
    user_answers = {}

    print("🧠 MBTI Personality Quiz")
    print("="*50)
    print("For each question, type 'A' or 'B' and press Enter.\n")

    for idx, row in df.iterrows():
        print(f"\nQ{idx+1}: {row['Question']}")
        print(f"  A) {row['Option A']}  → ({row['Option A Trait']})")
        print(f"  B) {row['Option B']}  → ({row['Option B Trait']})")

        while True:
            choice = input("Your choice (A/B): ").strip().upper()
            if choice in ['A', 'B']:
                break
            print("❌ Please enter 'A' or 'B'")

        # Record answer
        selected_option = 'A' if choice == 'A' else 'B'
        selected_trait = row[f'Option {selected_option} Trait']
        scores[selected_trait] += 1
        user_answers[idx+1] = selected_option

    # Calculate MBTI Type
    mbti = ""
    mbti += "E" if scores["E"] >= scores["I"] else "I"
    mbti += "S" if scores["S"] >= scores["N"] else "N"
    mbti += "T" if scores["T"] >= scores["F"] else "F"
    mbti += "J" if scores["J"] >= scores["P"] else "P"

    print("\n" + "="*50)
    print(f"🎉 Your MBTI Personality Type: {mbti}")
    print(f"📊 Raw Scores: {scores}")

    return mbti, user_answers, scores

import pandas as pd

def get_mbti_profile(mbti_type, csv_path='personality_description.csv'):

    try:
        # Load the CSV
        df = pd.read_csv(csv_path)

        # Normalize input (strip spaces, uppercase)
        mbti_type = mbti_type.strip().upper()

        # Find matching row
        profile_row = df[df['Personality Type'] == mbti_type]

        if profile_row.empty:
            print(f"❌ MBTI type '{mbti_type}' not found in database.")
            return None

        # Convert to dictionary
        profile = profile_row.iloc[0].to_dict()

        return profile

    except FileNotFoundError:
        print(f"❌ File '{csv_path}' not found. Please check the path.")
        return None
    except Exception as e:
        print(f"❌ Error loading profile: {e}")
        return None
    
from huggingface_hub import notebook_login

notebook_login()
