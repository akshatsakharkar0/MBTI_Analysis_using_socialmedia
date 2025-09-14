from flask import Flask, render_template, request, jsonify
import os
import joblib
import json
from datetime import datetime
import pandas as pd
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

#pip install -r requirement.txt
#python app.py

app = Flask(__name__, template_folder='templates')

# Load Hugging Face API
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    client = InferenceClient(token=HF_TOKEN)
else:
    client = None

# Load the pipeline and label encoders when the app starts
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')
dataset_dir = os.path.join(base_dir, 'dataset')
dataset_dir = os.path.join(base_dir, 'dataset')

# Load pipeline
pipeline_path = os.path.join(models_dir, 'pipeline.pkl')
pipeline = joblib.load(pipeline_path)

# Load all label encoders
label_encoders = {}
for dim in ['EI', 'NS', 'FT', 'JP']:
    le_path = os.path.join(models_dir, f'{dim}_label_encoder.pkl')
    label_encoders[dim] = joblib.load(le_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    message = data.get('message', '')
    mbti = data.get('mbti', 'INFP')
    
    if not message:
        return jsonify({'error': 'Please enter a message'}), 400
    
    # Get personality context
    try:
        csv_path = os.path.join(dataset_dir, 'personality_description.csv')
        df = pd.read_csv(csv_path)
        profile_row = df[df['Personality Type'] == mbti]
        if not profile_row.empty:
            profile = profile_row.iloc[0]
            context = f"{profile['Title']} with strengths: {profile['Strengths']}"
        else:
            context = mbti
    except:
        context = mbti
    
    # Try Hugging Face API first
    if client and HF_TOKEN:
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"You are an empathetic MBTI counselor. The user has {mbti} personality type. Provide personalized, supportive advice based on their personality traits. Keep responses concise and helpful."
                },
                {
                    "role": "user", 
                    "content": message
                }
            ]
            
            response = client.chat_completion(
                messages=messages,
                model="Qwen/Qwen2.5-72B-Instruct",
                max_tokens=150,
                temperature=0.7
            )
            
            if response and response.choices and len(response.choices) > 0:
                reply = response.choices[0].message.content.strip()
                if reply and len(reply) > 5:
                    return jsonify({'reply': reply})
        except Exception as e:
            print(f"HF API error: {e}")
            # Continue to fallback
    
    # Create context-aware responses based on user message
    if 'stress' in message.lower() or 'anxious' in message.lower():
        stress_responses = {
            'INFP': f"I understand you're feeling stressed. As an INFP, you might find peace through creative expression or connecting with your core values. What usually helps you feel centered?",
            'INFJ': f"Stress can be overwhelming for INFJs. Your intuition is telling you something important. Try some quiet reflection - what does your inner voice suggest?",
            'ENFP': f"ENFPs like you can feel stressed when possibilities feel limited. Remember, you're incredibly adaptable! What new perspective could shift this situation?",
            'ENFJ': f"As an ENFJ, you might be carrying others' stress too. It's okay to focus on yourself right now. What support do you need?"
        }
        if mbti in stress_responses:
            return jsonify({'reply': stress_responses[mbti]})
    
    elif 'relationship' in message.lower() or 'friend' in message.lower():
        relationship_responses = {
            'INFP': f"Relationships matter deeply to INFPs. Trust your values and communicate authentically. What feels most important to preserve here?",
            'ENFJ': f"Your ENFJ nature makes you a natural relationship builder. How can you create harmony while staying true to yourself?",
            'ISFJ': f"ISFJs like you are incredibly loyal friends. Remember that healthy boundaries actually strengthen relationships. What boundaries might help?"
        }
        if mbti in relationship_responses:
            return jsonify({'reply': relationship_responses[mbti]})
    
    elif 'work' in message.lower() or 'career' in message.lower():
        work_responses = {
            'INTJ': f"INTJs excel at strategic career planning. What's your long-term vision, and what steps align with that goal?",
            'ENTJ': f"Your ENTJ leadership skills are valuable in any career. What impact do you want to make, and how can you position yourself for that?",
            'ISFP': f"Work should align with your ISFP values. What kind of work environment would let your authentic self shine?"
        }
        if mbti in work_responses:
            return jsonify({'reply': work_responses[mbti]})
    
    # Simple greeting responses
    if message.lower() in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']:
        greeting_responses = {
            'INFP': "Hello! As an INFP, you bring such authenticity to the world. How are you feeling today?",
            'INFJ': "Hi there! Your INFJ intuition is a gift. What's on your mind today?",
            'ENFP': "Hey! Your ENFP energy always brightens the day. What exciting things are happening?",
            'ENFJ': "Hello! As an ENFJ, you naturally care for others. How can I support you today?",
            'INTP': "Hi! Your INTP mind loves exploring ideas. What's got you curious lately?",
            'INTJ': "Hello! INTJs like you see possibilities others miss. What are you working toward?",
            'ENTP': "Hey! Your ENTP creativity is amazing. What new ideas are you exploring?",
            'ENTJ': "Hello! As an ENTJ leader, you're always moving forward. What goals are you tackling?"
        }
        reply = greeting_responses.get(mbti, "Hello! I'm here to help you navigate life with your unique personality strengths. What's on your mind?")
        return jsonify({'reply': reply})
    
    # Default personalized responses without repeating the message
    default_responses = {
        'INFP': "As an INFP, your values guide your decisions. What feels most important to you right now?",
        'INFJ': "Your INFJ intuition often knows the answer before your mind does. What is it telling you?",
        'ENFP': "Your ENFP enthusiasm can turn any challenge into an opportunity! What possibilities do you see?",
        'ENFJ': "ENFJs like you naturally inspire others. How can you approach this with both heart and wisdom?",
        'INTP': "Your INTP analytical mind excels at breaking down complex issues. What patterns are you noticing?",
        'INTJ': "INTJs see the strategic big picture. What's your long-term vision here?",
        'ENTP': "Your ENTP innovation thrives on challenges. What creative solutions come to mind?",
        'ENTJ': "As an ENTJ, you're built for decisive action. What's your plan to move forward?",
        'ISFP': "Your ISFP authenticity is your superpower. What approach feels most true to you?",
        'ISTP': "Your ISTP practicality cuts through complexity. What hands-on solution makes sense?",
        'ESFP': "Your ESFP warmth brings joy to others. How can you channel that positive energy?",
        'ESTP': "ESTPs like you excel at bold, immediate action. What step can you take right now?",
        'ISFJ': "Your ISFJ caring nature is a true gift. How can you help while taking care of yourself?",
        'ISTJ': "Your ISTJ reliability creates stability for everyone. What systematic approach would work?",
        'ESFJ': "Your ESFJ social skills naturally bring people together. How can you create harmony?",
        'ESTJ': "As an ESTJ, you excel at organizing and executing. What structure would help solve this?"
    }
    
    reply = default_responses.get(mbti, "I understand what you're going through. Can you tell me more about your situation?")
    return jsonify({'reply': reply})

@app.route('/quiz_data')
def quiz_data():
    csv_path = os.path.join(dataset_dir, 'mbti_questions_dataset.csv')
    df = pd.read_csv(csv_path)
    questions = df.to_dict(orient='records')
    return jsonify({"questions": questions})

@app.route('/save_quiz_result', methods=['POST'])
def save_quiz_result():
    data = request.get_json()
    results_file = os.path.join(base_dir, 'quiz_results.json')
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    
    results.append({
        'name': data.get('name'),
        'email': data.get('email'),
        'mbti': data.get('mbti'),
        'timestamp': datetime.now().isoformat()
    })
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return jsonify({'success': True})

@app.route('/profile/<mbti_type>')
def profile(mbti_type):
    csv_path = os.path.join(dataset_dir, 'personality_description.csv')
    try:
        df = pd.read_csv(csv_path)
        mbti_type = mbti_type.strip().upper()
        profile_row = df[df['Personality Type'] == mbti_type]
        if profile_row.empty:
            return "MBTI type not found", 404
        profile = profile_row.iloc[0].to_dict()
        return render_template('profile.html', profile=profile, mbti_type=mbti_type)
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text_input')
    if not text:
        return jsonify({'error': 'Please enter text'}), 400

    # Predict classes
    prediction_nums = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])

    result = {}
    mbti = ""
    for idx, dim in enumerate(['EI', 'NS', 'FT', 'JP']):
        pred_num = prediction_nums[idx]
        le = label_encoders[dim]
        pred_label = le.inverse_transform([pred_num])[0]
        
        prob_array = probabilities[idx][0]
        confidence = round(prob_array[pred_num] * 100, 2)

        result[dim] = {
            'prediction': pred_label,
            'confidence': confidence
        }
        mbti += pred_label

    return jsonify({
        'mbti': mbti,
        'details': result
    })

if __name__ == "__main__":
    app.run(debug=True)