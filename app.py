```python
import joblib
import pandas as pd
import gradio as gr

# Load the trained ML model
model = joblib.load("student_marks_pipeline.pkl")


# Predict student marks
def predict_marks(study_hours):
    new_student = pd.DataFrame({
        "study_hours": [study_hours]
    })

    predicted_marks = model.predict(new_student)

    return (
        f"Study Hours: {study_hours}\n"
        f"Predicted Marks: {predicted_marks[0]:.2f} / 100"
    )


# Create the Gradio interface
demo = gr.Interface(
    fn=predict_marks,
    inputs=gr.Slider(
        minimum=0,
        maximum=24,
        step=0.5,
        value=5,
        label="Enter Study Hours"
    ),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Student Marks Prediction",
    description="Enter your study hours to predict your marks."
)

if __name__ == "__main__":
    demo.launch()
```
