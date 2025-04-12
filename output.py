from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk.tokenize import word_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from diffusers import AutoPipelineForText2Image
from transformers import pipeline
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from packaging import version
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import math
import os
import warnings
import streamlit as st
import nltk
nltk.download('punkt')


### TEXT SUMMARIZATION 


# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def calculate_metrics(reference, prediction):
    """Calculate classification metrics between reference and prediction texts"""
    ref_tokens = set(reference.lower().split())
    pred_tokens = set(prediction.lower().split())
    
    # Convert tokens to binary vectors
    all_tokens = list(ref_tokens.union(pred_tokens))
    ref_vector = [1 if token in ref_tokens else 0 for token in all_tokens]
    pred_vector = [1 if token in pred_tokens else 0 for token in all_tokens]
    
    return {
        "accuracy": accuracy_score(ref_vector, pred_vector),
        "precision": precision_score(ref_vector, pred_vector),
        "recall": recall_score(ref_vector, pred_vector),
        "f1": f1_score(ref_vector, pred_vector)
    }

### NEXT WORD PREDICTION

@st.cache_resource
def load_gpt2():
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), tokenizer, device

# Predict next few words using GPT-2 [5][6]
def predict_next_words(prompt, model, tokenizer, device, num_words=2):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + num_words,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_words = generated_text[len(prompt):].split()[:num_words]
    
    return " ".join(predicted_words)

# Calculate metrics for predicted words [3][5]
def calculate_metrics_next(predicted_words, reference_words):
    predicted_tokens = predicted_words.lower().split()
    reference_tokens = reference_words.lower().split()
    
    # Create binary labels (1 if word appears in reference, else 0)
    y_true = [1 if word in reference_tokens else 0 for word in predicted_tokens]
    y_pred = [1] * len(predicted_tokens)  # Assume all predicted words are positive
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    
    return accuracy, precision, recall, f1

# Perplexity calculation for context + prediction [2][5]
def calculate_perplexity(prompt, predicted_words, model, tokenizer, device):
    full_text = prompt + " " + predicted_words
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    return torch.exp(outputs.loss).item()


### STORY PREDICTION 

@st.cache_resource
# Predict the continuation of a story
def predict_story(prompt, model, tokenizer, device, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=len(inputs.input_ids[0]) + max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.6,  # Lower temperature for more coherent text
            num_beams=4,      # Beam search for better coherence
            no_repeat_ngram_size=3,  # Prevent repetition
            pad_token_id=tokenizer.pad_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()

# Calculate evaluation metrics
def calculate_metrics_story(predicted_text, reference_text):
    predicted_tokens = predicted_text.lower().split()
    reference_tokens = reference_text.lower().split()
    
    y_true = [1 if word in reference_tokens else 0 for word in predicted_tokens]
    y_pred = [1] * len(predicted_tokens)  # Assume all predicted words are positive
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    
    return accuracy, precision, recall, f1

### CHATBOT INTERACTION

# Load Blenderbot-400M-Distill model with caching for performance
@st.cache_resource
def load_chatbot():
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# Initialize the chatbot
tokenizer, model = load_chatbot()

# Initialize conversation history in Streamlit session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Function to generate chatbot response
def generate_response(user_input):
    try:
        # Tokenize user input and generate response
        inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True)
        reply_ids = model.generate(
            inputs.input_ids,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        response = tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()
        return response
    except Exception as e:
        return f"Error: {str(e)}"

### SENTIMENT ANALYSIS

# Load model and tokenizer with caching for performance
@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Initialize the model and tokenizer
tokenizer, model = load_sentiment_model()

# Sentiment prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        sentiment_map = {0: "Negative", 1: "Positive"}
        return sentiment_map[predicted_class], probabilities[0][predicted_class].item()

# Metrics calculation function
def calculate_metrics_sentiment(predictions, references):
    y_true = [1 if ref.lower() == "positive" else 0 for ref in references]
    y_pred = [1 if pred.lower() == "positive" else 0 for pred in predictions]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    
    return accuracy, precision, recall, f1

### IMAGE GENERATION

# Load Stable Diffusion 2.1 model with caching
@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        revision="fp16" if torch.cuda.is_available() else None
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

# Initialize the pipeline
pipe = load_model()


# Apply CSS for full-width tabs
st.markdown("""<style>
        div[data-testid="stTabs"] button {
            flex-grow: 1;
            justify-content: center;
        }
    </style>""", unsafe_allow_html=True)

# Streamlit app title
st.title("MULTIFUNCTIONAL NLP AND IMAGE GENERATION TOOL USING HUGGING FACE MODELS")

with st.sidebar:
    st.title(":red[HOME]")
    st.header("PROBLEM STATEMENT")

# Create six tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["TEXT SUMMARIZATION", "NEXT WORD PREDICTION", "STORY PREDICTION", "CHATBOT", "SENTIMENT ANALYSIS",
                                               "IMAGE GENERATION"])


# Content for each tab
with tab1:
    st.header("TEXT SUMMARIZATION")

    # Streamlit UI
    st.title("Text Summarization Analyzer")
    input_text = st.text_area("Input Text", height=200)
    reference_summary = st.text_area("Reference Summary (for evaluation)", height=100)

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            # Generate summary
            summary_result = summarizer(
                input_text, 
                max_length=150, 
                min_length=40, 
                do_sample=False
            )[0]['summary_text']
            
            # Calculate metrics
            metrics = calculate_metrics(reference_summary, summary_result)
            
        # Display results
        st.subheader("Generated Summary")
        st.write(summary_result)
        
        st.subheader("Evaluation Metrics")
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
        cols[1].metric("Precision", f"{metrics['precision']:.2%}")
        cols[2].metric("Recall", f"{metrics['recall']:.2%}")
        cols[3].metric("F1 Score", f"{metrics['f1']:.2%}")


with tab2:
    st.title("GPT-2 Large Next Few Words Predictor")
    model, tokenizer, device = load_gpt2()

    with st.form("prediction_form"):
        prompt = st.text_input("Input Prompt", "Artificial intelligence")
        reference_words = st.text_input("Expected Next Words", "is changing the world")
        num_words = st.select_slider("Number of Words to Predict", options=[2, 3, 4, 5, 6], value=3)
        submitted = st.form_submit_button("Predict Next Words")

    if submitted:
        with st.spinner("Generating prediction..."):
            try:
                # Generate prediction
                predicted_words = predict_next_words(prompt, model, tokenizer, device, num_words)
                accuracy, precision, recall, f1 = calculate_metrics_next(predicted_words, reference_words)
                perplexity = calculate_perplexity(prompt, predicted_words, model, tokenizer, device)

                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Results")
                    st.markdown(f"**Input Context:** `{prompt}`")
                    st.markdown(f"**Predicted Words:** `{predicted_words}`")
                    st.markdown(f"**Expected Words:** `{reference_words}`")
                    st.metric("Perplexity", f"{perplexity:.2f}")
                    
                with col2:
                    st.subheader("Evaluation Metrics")
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    st.metric("Precision", f"{precision*100:.2f}%")
                    st.metric("Recall", f"{recall*100:.2f}%")
                    st.metric("F1 Score", f"{f1*100:.2f}%")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")


with tab3:
    st.title("GPT-2 Large Story Prediction")
    model, tokenizer, device = load_gpt2()

    with st.form("story_form"):
        prompt = st.text_area("Input Story Start", "Once upon a time in a distant land, there was a...")
        reference_text = st.text_area("Expected Continuation (for evaluation)", "young prince who dreamed of exploring the world.")
        max_length = st.slider("Max Length of Predicted Story", min_value=50, max_value=300, value=150)
        submitted = st.form_submit_button("Predict Story")

    if submitted:
        with st.spinner("Generating story..."):
            try:
                # Generate story continuation
                predicted_story = predict_story(prompt, model, tokenizer, device, max_length)
                
                # Evaluate metrics
                accuracy, precision, recall, f1 = calculate_metrics_story(predicted_story, reference_text)

                # Display results
                st.subheader("Predicted Story Continuation")
                st.write(predicted_story)

                st.subheader("Evaluation Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    st.metric("Precision", f"{precision*100:.2f}%")
                
                with col2:
                    st.metric("Recall", f"{recall*100:.2f}%")
                    st.metric("F1 Score", f"{f1*100:.2f}%")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
       

with tab4:
    # Function to rerun the app based on Streamlit version
    def rerun_app():
        if version.parse(st.__version__) >= version.parse("1.27.0"):
            st.rerun()  # Use st.rerun for newer versions
        else:
            st.experimental_rerun()  # Use st.experimental_rerun for older versions

    # Streamlit UI setup
    st.title("🤖 Chatbot with Blenderbot-400M-Distill")
    st.write("Ask me anything! I'm here to chat with you.")

    # Display conversation history
    if st.session_state.conversation:
        for i, message in enumerate(st.session_state.conversation):
            if i % 2 == 0:  # User messages (even indices)
                st.markdown(f"**You:** {message}")
            else:  # Chatbot responses (odd indices)
                st.markdown(f"**Chatbot:** {message}")

    # User input field
    user_input = st.text_input("Your message:", key="user_input")

    # Handle user input and generate response
    if st.button("Send"):
        if user_input.strip():  # Ensure input is not empty
            # Add user message to conversation history
            st.session_state.conversation.append(user_input)

            # Generate chatbot response and add to conversation history
            with st.spinner("Chatbot is thinking..."):
                bot_response = generate_response(user_input)
                st.session_state.conversation.append(bot_response)

            # Clear the input field after submission and rerun the app
            rerun_app()

    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        rerun_app()
    


with tab5:
    # Streamlit UI setup
    st.title("Sentiment Analysis with DistilBERT")
    st.write("Provide sentences to predict their sentiment (Positive/Negative) and evaluate the model.")

    # Input text area for sentences
    user_inputs = st.text_area("Enter sentences (one per line):", placeholder="E.g., I love this product!\nThis is the worst experience ever.")

    # Reference sentiments for evaluation (optional)
    reference_sentiments = st.text_area("Enter reference sentiments (one per line):", placeholder="E.g., Positive\nNegative")

    if st.button("Analyze Sentiment"):
        if user_inputs.strip():
            # Split user inputs into individual sentences
            sentences = user_inputs.strip().split("\n")
            
            # Predict sentiments for each sentence
            predictions = []
            confidences = []
            for sentence in sentences:
                sentiment, confidence = predict_sentiment(sentence)
                predictions.append(sentiment)
                confidences.append(confidence)
            
            # Display predictions and confidences
            st.subheader("Predictions")
            for i, sentence in enumerate(sentences):
                st.write(f"**Sentence:** {sentence}")
                st.write(f"**Predicted Sentiment:** {predictions[i]} (Confidence: {confidences[i]:.2f})")
            
            # If reference sentiments are provided, calculate metrics
            if reference_sentiments.strip():
                references = reference_sentiments.strip().split("\n")
                if len(references) == len(sentences):
                    accuracy, precision, recall, f1 = calculate_metrics_sentiment(predictions, references)
                    
                    # Display evaluation metrics
                    st.subheader("Evaluation Metrics")
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    st.metric("Precision", f"{precision*100:.2f}%")
                    st.metric("Recall", f"{recall*100:.2f}%")
                    st.metric("F1 Score", f"{f1*100:.2f}%")
                else:
                    st.error("The number of reference sentiments does not match the number of input sentences.")
        else:
            st.error("Please enter at least one sentence.")

    if st.button("Clear"):
        st.session_state.clear()
    


with tab6:
    # Streamlit UI
    st.title("Stable Diffusion 2.1 Image Generator")

    # User input
    prompt = st.text_input("Enter your prompt:", "A futuristic cityscape at sunset")

    if st.button("Generate Image"):
        if prompt:
            with st.spinner("Generating image..."):
                try:
                    # Generate image
                    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        width=512,
                        height=512,
                        generator=generator
                    ).images[0]

                    # Display image
                    st.image(image, caption=prompt, use_container_width=True)  # Updated parameter
                    
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")
        else:
            st.warning("Please enter a prompt")
    





