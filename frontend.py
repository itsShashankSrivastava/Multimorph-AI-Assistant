# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()


#Step1: Setup UI with streamlit (model provider, model, system prompt, web_search, query)
import streamlit as st

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")

st.markdown(
    """
    <style>
    .rounded-image {
        border-radius: 50%;
        width: 150px;  # Adjust the size of the image as needed
        height: 150px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add the image with the custom CSS class
st.markdown('<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRVqAOZRUkZgZMc_kqNTAOGEee8PPywg2aWBHU5rFheDS3rL0mzc7foF3ahWrfFDOgyzfg&usqp=CAU" class="rounded-image">', unsafe_allow_html=True)


st.title("MultiMorph: Your Anytime Assistant")
st.write("Create and Interact with the AI Agents!")

system_prompt=st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_ANTHROPIC = ["claude-3-5-sonnet-20241022"]

provider=st.radio("Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "Anthropic":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_ANTHROPIC)

allow_web_search=st.checkbox("Allow Web Search")

user_query=st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

API_URL="http://127.0.0.1:9999/chat"

if st.button("Ask Agent!"):
    if user_query.strip():
        #Step2: Connect with backend via URL
        import requests

        payload={
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        response=requests.post(API_URL, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response_data}")