"""Verifies app.py actually runs — every page, and the Predict button
with a real form submission — not just that it imports cleanly."""

from streamlit.testing.v1 import AppTest


def run_page(page_label):
    at = AppTest.from_file("app.py")
    at.run(timeout=60)
    assert not at.exception, f"Exception on initial load: {at.exception}"
    at.sidebar.radio[0].set_value(page_label).run(timeout=60)
    assert not at.exception, f"Exception on page '{page_label}': {at.exception}"
    print(f"OK: page '{page_label}' rendered with no exception.")
    return at


for label in ["Home", "Data Exploration", "Model Performance", "Predict Attrition"]:
    run_page(label)

# Now actually submit the predict form and confirm we get a real prediction.
at = AppTest.from_file("app.py")
at.run(timeout=60)
at.sidebar.radio[0].set_value("Predict Attrition").run(timeout=60)
assert not at.exception, f"Exception before submit: {at.exception}"

# Set OverTime to Yes to exercise the high-risk driver path.
selectboxes = at.selectbox
overtime_box = None
for sb in selectboxes:
    if sb.label == "OverTime":
        overtime_box = sb
        break
assert overtime_box is not None, "Could not find OverTime selectbox"
overtime_box.set_value("Yes")

at.button[0].click().run(timeout=60)
assert not at.exception, f"Exception after submit: {at.exception}"

md_texts = " ".join(m.value for m in at.markdown)
assert "Predicted attrition probability" in md_texts
assert "Risk category" in md_texts
assert "Key drivers" in md_texts
print("OK: Predict button submitted successfully and produced a real prediction + risk category + drivers.")

print("\nALL APP TESTS PASSED")
