from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "MedQuAD"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = PROCESSED_DATA_DIR / "clinical_knowledge.csv"


# --------------------------------------------------
# XML Parser
# --------------------------------------------------

def parse_xml_file(xml_file):
    """
    Parse one MedQuAD XML file and return
    a list of question-answer records.
    """

    records = []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        document_id = root.attrib.get("id")
        source = root.attrib.get("source")
        url = root.attrib.get("url")

        # Focus
        focus_element = root.find("Focus")
        focus = (
            focus_element.text.strip()
            if focus_element is not None and focus_element.text
            else None
        )

        # Category
        category_element = root.find(
            "FocusAnnotations/Category"
        )

        category = (
            category_element.text.strip()
            if category_element is not None
            and category_element.text
            else None
        )

        # UMLS CUI
        cui_element = root.find(
            "FocusAnnotations/UMLS/CUIs/CUI"
        )

        umls_cui = (
            cui_element.text.strip()
            if cui_element is not None
            and cui_element.text
            else None
        )

        # Semantic group
        semantic_group_element = root.find(
            "FocusAnnotations/UMLS/SemanticGroup"
        )

        semantic_group = (
            semantic_group_element.text.strip()
            if semantic_group_element is not None
            and semantic_group_element.text
            else None
        )

        # Synonyms
        synonyms = []

        for synonym in root.findall(
            "FocusAnnotations/Synonyms/Synonym"
        ):
            if synonym.text:
                synonyms.append(synonym.text.strip())

        synonyms_text = " | ".join(synonyms)

        # Questions / Answers
        for qa in root.findall("QAPairs/QAPair"):

            question_element = qa.find("Question")
            answer_element = qa.find("Answer")

            if question_element is None:
                continue

            question = (
                question_element.text.strip()
                if question_element.text
                else ""
            )

            answer = (
                answer_element.text.strip()
                if answer_element is not None
                and answer_element.text
                else ""
            )

            # Ignore empty answers
            if not answer:
                continue

            qid = question_element.attrib.get("qid")
            qtype = question_element.attrib.get("qtype")

            records.append(
                {
                    "document_id": document_id,
                    "qid": qid,
                    "source": source,
                    "url": url,
                    "focus": focus,
                    "category": category,
                    "umls_cui": umls_cui,
                    "semantic_group": semantic_group,
                    "synonyms": synonyms_text,
                    "question": question,
                    "question_type": qtype,
                    "answer": answer,
                }
            )

    except ET.ParseError as e:
        print(f"XML parsing error: {xml_file}")
        print(e)

    except Exception as e:
        print(f"Error processing {xml_file}")
        print(e)

    return records


# --------------------------------------------------
# Process entire dataset
# --------------------------------------------------

def build_dataset():

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    xml_files = list(
        RAW_DATA_DIR.rglob("*.xml")
    )

    print(
        f"Found {len(xml_files)} XML files."
    )

    all_records = []

    for index, xml_file in enumerate(xml_files):

        records = parse_xml_file(xml_file)

        all_records.extend(records)

        if (index + 1) % 1000 == 0:
            print(
                f"Processed {index + 1}/{len(xml_files)} files..."
            )

    df = pd.DataFrame(all_records)

    print("\nBefore cleaning:")
    print(f"Records: {len(df)}")

    # Remove duplicate questions
    df = df.drop_duplicates(
        subset=["question", "answer"]
    )

    # Remove rows without question/answer
    df = df.dropna(
        subset=["question", "answer"]
    )

    df = df[
        (df["question"].str.strip() != "")
        & (df["answer"].str.strip() != "")
    ]

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\nAfter cleaning:")
    print(f"Records: {len(df)}")

    print(
        f"\nSaved dataset to:\n{OUTPUT_FILE}"
    )

    return df


if __name__ == "__main__":
    build_dataset()