from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from lib.utils import PROMPT_PATH


from dotenv import load_dotenv
load_dotenv()

GEMINI_FLASH_MODEL="google_genai:gemini-3-flash-preview"
GROQ_MODEL_STRING="moonshotai/kimi-k2-instruct-0905"

# llm = init_chat_model(GEMINI_FLASH_MODEL)
llm = ChatGroq(
    model=GROQ_MODEL_STRING,
    temperature=0,
    max_tokens=None,
    # reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)


def chat(prompt: str,verbose=False):
    result = llm.invoke(prompt)
    if isinstance(result, str):
        return result
    elif isinstance(result, AIMessage): 
        token_usage = result.response_metadata["token_usage"]
        if verbose:
            return f"{result.content}\nCompletion tokens: {token_usage["completion_tokens"]}\nPrompt tokens:{token_usage["prompt_tokens"]}\nTotal tokens:{token_usage["total_tokens"]}"
        else:
            return result.content

def spell_check(text):
    prompt = None
    with open(PROMPT_PATH / "spell_check.md") as f:
        prompt = f.read()
    prompt = prompt.format(query=text)
    result = chat(prompt)
    return result

def test(text):
    print(spell_check(text))