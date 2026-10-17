
import streamlit as st
from src.rag_pipeline import answer_question
import logging

# 1. Configure the page
st.set_page_config(
    page_title="ClinRAG",
    page_icon="🩺",
    layout="wide"
)


# 2. Display the application header
st.markdown(
    """
    <div style="
        padding: 1.5rem 1.75rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #111C33, #1D355F);
        border-left: 5px solid #3B82F6;
        margin-bottom: 1.25rem;
    ">
        <div style="
            color: #93C5FD;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 2px;
            text-transform: uppercase;
        ">
            AI Research Project
        </div>
        <h1 style="
            color: #FFFFFF;
            font-size: 2.5rem;
            margin: 0.4rem 0;
        ">
            ClinRAG
        </h1>
        <p style="
            color: #D7E3F4;
            font-size: 1.05rem;
            margin: 0;
        ">
            Evidence-Grounded Clinical Knowledge Retrieval
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <p style="
        font-size: 1.05rem;
        color: #64748B;
        margin: 0.5rem 0 1.25rem 0;
    ">
        Explore clinical questions with AI-generated answers
        grounded in evidence retrieved from the MedQuAD dataset.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        padding: 0.85rem 1rem;
        border: 1px solid #F59E0B;
        border-radius: 10px;
        background-color: rgba(245, 158, 11, 0.08);
        margin-bottom: 1.5rem;
    ">
        <span style="color: #D97706; font-weight: 700;">
            Research prototype
        </span>
        <span style="color: inherit;">
            &nbsp; Not intended for diagnosis or personalized
            medical advice. Verify clinical information with
            qualified healthcare professionals.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)




# 3. Accept the user's question
st.markdown(
    """
    <h3 style="margin-bottom: 0.25rem;">
        Ask a clinical question
    </h3>
    <p style="
        color: #64748B;
        font-size: 0.95rem;
        margin-top: 0;
    ">
        Enter your question to retrieve relevant evidence
        and generate an evidence-grounded answer.
    </p>
    """,
    unsafe_allow_html=True,
)

question = st.text_area(
    "Your question",
    placeholder="e.g., What are the symptoms of diabetes?",
    height=120,
    max_chars=500,
    help="Enter a clinical knowledge question in plain language.",
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
            logging.exception("ClinRAG request failed")
            st.error("The question could not be processed.")
            st.info(
                "Please try again shortly. If the problem "
                "continues, the service may be temporarily unavailable."
            )
