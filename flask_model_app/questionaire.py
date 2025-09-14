import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'dataset', 'mbti_questions_dataset.csv')
df = pd.read_csv(csv_path)

def run_mbti_quiz(df):
    # Initialize scores for each trait
    scores = {"E": 0, "I": 0, "S": 0, "N": 0, "T": 0, "F": 0, "J": 0, "P": 0}
    user_answers = {}

    print("🧠 Welcome to the MBTI Personality Quiz!")
    print("="*50)
    print("For each question, type 'A' or 'B' and press Enter.\n")

    for idx, row in df.iterrows():
        print(f"\nQ{idx+1}: {row['Question']}")
        print(f"  A) {row['Option A']}  → ({row['Option A Trait']})")
        print(f"  B) {row['Option B']}  → ({row['Option B Trait']})")

        # Get user input and validate
        while True:
            choice = input("Your choice (A/B): ").strip().upper()
            if choice in ['A', 'B']:
                break
            print("❌ Please enter 'A' or 'B'")

        # Record the user's choice
        selected_option = 'A' if choice == 'A' else 'B'
        selected_trait = row[f'Option {selected_option} Trait']
        scores[selected_trait] += 1
        user_answers[idx+1] = selected_option

    # Determine final MBTI type by comparing scores
    mbti = ""
    mbti += "E" if scores["E"] >= scores["I"] else "I"
    mbti += "S" if scores["S"] >= scores["N"] else "N"
    mbti += "T" if scores["T"] >= scores["F"] else "F"
    mbti += "J" if scores["J"] >= scores["P"] else "P"

    print("\n" + "="*50)
    print(f"🎉 Your MBTI Personality Type is: {mbti}")
    print("📊 Here are your scores:")
    for trait, score in scores.items():
        print(f"   {trait}: {score}")

    return mbti, user_answers, scores

if __name__ == "__main__":
    # Run the quiz
    run_mbti_quiz(df)
