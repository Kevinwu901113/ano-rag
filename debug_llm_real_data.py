import requests
import json

def check_llm_on_data():
    # Manually checking the first few questions that failed
    cases = [
        {
            "question": "What is John Mayne's occupation?",
            "context": """[1] John Dawson Mayne
Mayne served as the Professor of law, logic and moral philosophy at the Presidency College, Madras from 1857 throughout the 1860s. He also served as Assistant Legal Secretary to the Madras government from 1860 to 1872 and as a Clerk of the Crown during the 1860s. He served as Advocate-General of Madras from 1862 to 1872. He left India in a cloud of scandal, running away from his wife with the wife of another man, Annie Craigie-Halkett. In England, despite the scandal, Mayne served as a Professor of Common Law at the Inns of Court from 1879 to 1883. In 1880, he unsuccessfully contested for the Parliamentary seat at Falmouth. He was an enthusiastic family historian, producing an impressively long 'pedigree' of the Maynes from 1900 back through some thirty generations to Normandy, but beyond the 17th century, like so many family histories of the time, it was riddled with errors of assumption. At Madras, 1859, he married his first wife, Helen Sarah Hamilton (born 1841), daughter of Colonel Robert Hamilton of the Madras Staff Corps.

[2] John Dawson Mayne
British lawyer (1828–1917) John Dawson Mayne (1828–1917) was a British lawyer and legal expert who served as acting Advocate-General of the Madras Presidency. He is remembered as the author of "Mayne's Hindu Law" regarded as the most authoritative book on the Indian Penal Code. His married life was marred by a scandal, which prevented him from gaining a knighthood. Family. Born on 31 December 1828, to John Mayne (1793–1828..."""
        }
    ]

    url = "http://127.0.0.1:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    prompt_template = """Answer the question based on the context below. Keep the answer short and concise. Do not output reasoning.

Context:
{context}

Question: {question}
Answer:"""

    for case in cases:
        prompt = prompt_template.format(context=case["context"], question=case["question"])
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 100
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            print(f"Question: {case['question']}")
            print(f"Raw Answer: {content!r}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_llm_on_data()
