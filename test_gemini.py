import os

from dotenv import load_dotenv

from gemini_service import create_client, generation_model


def main():
    """Manual live-API check; importing this file never sends a request."""
    print("เริ่มโปรแกรม")
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY_INSURVERSE")
    if not api_key:
        print("ไม่พบ GEMINI_API_KEY ในไฟล์ .env")
        return 1

    print("พบ API Key แล้ว")
    client = create_client(api_key)
    print("กำลังส่งคำถามไป Gemini...")
    try:
        response = client.models.generate_content(
            model=generation_model(),
            contents="สวัสดี ช่วยแนะนำตัวสั้น ๆ",
        )
        print("คำตอบจาก Gemini:")
        print(response.text)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
