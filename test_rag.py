from query_data import query_rag
from langchain_ollama import OllamaLLM as Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_monopoly_rules():
    assert query_and_validate(
        question="what is the CGPA of the student?",
        expected_response="3.23",
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="what is COMP2101 about or the course description?",
        expected_response="This course introduces some fundamental topics in computer science. This includes numbering systems, data representation, problem solving and algorithm design. Furthermore, the course includes the study and practice of basic programing concepts such as data types, variables, arrays, selection, repetition, data files and functions.",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )
    print('&'*20)
    print(response_text)
    print('&'*20)
    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
