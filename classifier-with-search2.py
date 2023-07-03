import os
import csv
import requests
import openai
import PyPDF2
import json


# Set up OpenAI API credentials
openai.api_key = "XXXXXXX"

# Folder path and output file path
folder_path = "/home/icat/openai/llm-doc-classification/data-set"
output_file = "/home/icat/openai/llm-doc-classification/classification-results.csv"

# Document types for classification
document_types = [
    "Offer to lease",
    "Waiver",
    "Service extension agreement",
    "Job displacement voucher",
    "Statement",
    "Asbestos managing guide",
    "Mortgage Brokerages, Lenders and Administrators Act",
    "Authority to Proceed",
    "Credit Card Authorization",
    "Trust Ledger Statement",
    "Due Dilligence Form",
    "Rules",
    "Policy",
    "Release of Information",
    "Certification of non-foreign status",
    "Memo",
    "Marketing Plan",
    "Cancellation Letter",
    "Job Offer",
    "Unemployment Notice",
    "Concent",
    "Health Assessment Form",
    "Authorization",
    "Deed poll",
    "Session Report",
    "Expense report",
    "Financial statement & Report",
    "Coverage rejection",
    "Billing information",
    "Payment Voucher",
    "Acknowledgement of Appraisal Delivery",
    "Notice",
    "Treatment plan progress",
    "Agreement",
    "Terms and conditions",
    "Amendment",
    "Pop up contract",
    "Invoice",
    "Rate Confirmation"
]

# Create the classification results CSV file and write the header
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Document Name", "Classified Document Type", "Classification Probability"])

# Iterate through PDF files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        
        # Print file name
        print(filename)
        
        # Read the first 1000 symbols from the PDF file
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()[:1000]
        
        classified_doc_type = None
        classification_prob = None

        # Send the request to the OpenAI API
        output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[
                {
                    "role": "user",
                    "content": "Take these first 1000 symbols from document contents as an input and based on the input, classify document into one of the given document types. Use only given document types for classification or 'other' if you can't classify it into one of the given types. Supplement your output with probability score. Present output in json format with the following keys - doctype, probability. Skip any other notes or comments, your output should contain json only. Input -> \n {}" + text + "\n Document types -> \n {}" + str(document_types)
                }
                ],
                temperature=0.3
            )

        # Parse the output JSON
        data = json.loads(str(output))

        # Extract the content from the message
        content = data["choices"][0]["message"]["content"]

        # Parse the content JSON
        content_data = json.loads(content)

        # Extract the doctype and probability
        classified_doc_type = content_data["doctype"]
        classification_prob = content_data["probability"]
        

        if classified_doc_type == "other":
            # Alternative code
            document_types = sorted(document_types, key=lambda x: len(x), reverse=True)
            for doc_type in document_types:
                if doc_type.lower() in text.lower():
                    classified_doc_type = doc_type
                    classification_prob = 0
                    print("Found!"+classified_doc_type)

        
    # Write the classification results to the CSV file
    with open(output_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([filename, classified_doc_type, classification_prob])
