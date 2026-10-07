from google import genai

client = genai.Client()
resp = client.models.generate_content(model="gemini-3.5-flash", contents="Say hi")
print(resp.text)