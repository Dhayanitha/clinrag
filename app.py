
import streamlit as st
from src.rag_pipeline import answer_question


# 1. Configure the page
st.set_page_config(
    page_title="ClinRAG",
    page_icon="🩺",
    layout="wide"
)


# 2. Display the application header
st.title("ClinRAG")
st.subheader("Evidence-Grounded Clinical Knowledge Retrieval")

st.write(
    "Ask a clinical knowledge question and explore the "
    "evidence retrieved from the MedQuAD dataset."
)

st.info(
    "Research prototype for clinical information retrieval. "
    "Not intended for diagnosis or personalized medical advice."
)


# 3. Accept the user's question
question = st.text_area(
    "Enter your question",
    placeholder="What are the symptoms of diabetes?",
    height=100,
    max_chars=500
)


# 4. Process the question
if st.button("Get Answer", type="primary"):
    cleaned_question = question.strip()

    if len(cleaned_question) < 3:
        st.warning("Please enter a question with at least 3 characters.")

    else:
        try:
            with st.spinner(
                "Retrieving evidence and generating your answer..."
            ):
                result = answer_question(cleaned_question)

            # 5. Display the generated answer
            st.subheader("Answer")
            st.write(result["answer"])

            # 6. Display source links
            st.subheader("Sources")

            if result.get("sources"):
                for source in result["sources"]:
                    source_name = source.get("source") or "Source"
                    source_url = source.get("url")

                    if source_url:
                        st.markdown(f"- [{source_name}]({source_url})")
                    else:
                        st.write(source_name)
            else:
                st.write("No source links were returned.")

            # 7. Display retrieved evidence
            st.subheader("Retrieved Evidence")

            if result.get("evidence"):
                for item in result["evidence"]:
                    title = (
                        f"Rank {item['rank']} | "
                        f"Score: {item['score']:.3f} | "
                        f"Source: {item.get('source') or 'Unknown'}"
                    )

                    with st.expander(title):
                        st.write(item["text"])

                        if item.get("qid"):
                            st.caption(f"Question ID: {item['qid']}")

                        if item.get("url"):
                            st.markdown(
                                f"[Open source]({item['url']})"
                            )
            else:
                st.write("No evidence was retrieved.")

        except Exception as error:
            st.error("The question could not be processed.")
            st.caption(
                "Check the VS Code terminal for the detailed error."
            )
            print(f"ClinRAG error: {error}")
