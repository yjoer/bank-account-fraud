from st_pages import Page
from st_pages import show_pages

show_pages(
    [
        Page("pages/evaluation_summary.py", "Model Evaluation"),
        Page("pages/evaluation_one.py", "Model Evaluation 1"),
        Page("pages/evaluation_two.py", "Model Evaluation 2"),
    ]
)
