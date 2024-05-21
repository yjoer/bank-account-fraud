from dotenv import load_dotenv
from st_pages import Page
from st_pages import show_pages

load_dotenv()

show_pages(
    [
        Page("pages/evaluation_summary.py", "Model Evaluation"),
        Page("pages/evaluation_one.py", "Model Evaluation 1"),
        Page("pages/evaluation_two.py", "Model Evaluation 2"),
        Page("pages/interpretation.py", "Model Interpretation"),
    ]
)
