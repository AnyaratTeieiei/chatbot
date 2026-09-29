# Hybrid Adaptive RAG — PELEK

## ไฟล์และการทำงาน
- `app.py`: คงหน้าตาเดิม เชื่อม routing, streaming, ประวัติ และแสดงเวลาใน expander
- `rag.py`: parser และ synonym เดิมร่วมกัน, TF-IDF ของ FAQ/คำถาม, confidence, hybrid ranking, metadata validation, cooldown, history และ streaming
- `build_embeddings.py`: สร้าง document embeddings แบบ batch เฉพาะเมื่อสั่งรัน
- `requirements.txt`: เพิ่ม numpy เป็น direct dependency
- `test_rag.py`, `test_app.py`: ทดสอบออฟไลน์และหน้า Streamlit โดยจำลอง Gemini

ทุกคำถามค้นหา TF-IDF ในเครื่องก่อน ใช้ expanded query เฉพาะการค้นหา และส่งคำถามต้นฉบับให้ Gemini

## คำถามต่อเนื่องและ Context Memory
- หลังตอบสำเร็จ ระบบจำหัวข้อ คำค้นแบบเต็ม และ FAQ สูงสุด 3 ข้อที่ใช้ โดยเก็บเฉพาะแหล่งข้อมูล FAQ ไม่ใช้ข้อความที่ Gemini เรียบเรียงเป็นข้อเท็จจริง
- ทุกคำถามค้นหาใหม่ก่อนเสมอ หากพบหัวข้อใหม่ชัดเจนหรือผลค้นหาใหม่มีความมั่นใจ ระบบจะไม่ใช้บริบทเก่า
- คำถามสั้นหรือคำอ้างอิง เช่น `อะไรบ้าง`, `แล้วต้องเลือกตอนไหน`, `อันแรก` จะรวมคำค้นก่อนหน้ากับคำถามล่าสุด เพิ่มน้ำหนัก FAQ ล่าสุด แล้วเลือก Top 3 ใหม่
- คำถามที่ระบุหัวข้อใหม่ เช่น `ค่าเทอมเท่าไหร่` จะตัดบริบทเดิมทันที
- ปุ่มล้างประวัติจะล้าง Context Memory และ embedding cooldown ด้วย

| Route | เงื่อนไข | การทำงาน | API calls |
|---|---|---|---|
| A | คะแนน >= 0.85 และส่วนต่าง >= 0.15 | คำตอบ FAQ ตรงพร้อมเลขอ้างอิง | 0 |
| B | คะแนน >= 0.22 และส่วนต่าง >= 0.04 | FAQ สูงสุด 3 ข้อให้ Gemini เรียบเรียง | 1 generate |
| C | คะแนนต่ำ/ใกล้กัน หรือคำถามอธิบาย/หลายประเด็น | query embedding หนึ่งครั้ง + hybrid + FAQ สูงสุด 3 ข้อให้ Gemini | 1 embedding + 1 generate |

คำถามหลายประเด็นตรวจจากคำบ่งชี้ก่อนพิจารณา A/B เกณฑ์เป็นค่าคงที่ใน `rag.py` และควรปรับด้วยชุดคำถามใช้งานจริง
หากไม่มี embeddings ที่ตรงข้อมูล หรืออยู่ใน cooldown: C กลับเป็น B (1 generate)
หาก query embedding ล้มเหลว: 1 embedding attempt + 1 generate และพัก embedding 60 วินาทีใน session นั้น
หากรูปแบบคำตอบ FAQ อ่านไม่ได้: A กลับเป็น B
หากไม่มี API key ยังใช้ A ได้ ส่วน B/C แจ้งให้ตั้งค่าระบบ

ประวัติส่งเฉพาะคู่ user/model ล่าสุด ไม่เกิน 6 ข้อความ คำถามล่าสุดส่งใน prompt ครั้งเดียว
SDK ที่รองรับ stream แสดงข้อความทยอยขึ้น; SDK เก่าใช้คำตอบปกติ โดยไม่ลองส่งซ้ำหลังเกิด network error
ไม่สร้าง document embeddings ตอนเปิดแอป ไม่แก้ FAQ และไม่เปลี่ยน model/SDK

## สร้าง embeddings เมื่อ FAQ เปลี่ยน
รันจากโฟลเดอร์ที่มี `app.py` โดยใช้ Python ที่ติดตั้ง dependencies:

```powershell
python -m pip install -r requirements.txt
python build_embeddings.py
python -m streamlit run app.py
```

คำสั่ง `build_embeddings.py` ใช้โควตา Gemini จริง อ่าน key จาก `GEMINI_API_KEY_INSURVERSE` ใน `.env` หรือ environment เท่านั้น
ตัวสร้างประมวลผลไม่เกิน 80 FAQ ต่อช่วง แล้วพัก 80 วินาทีก่อนทำรายการที่เหลือ หาก Gemini ตอบกลับด้วยโควตาต่อนาที (`429 ResourceExhausted`) ระบบจะพักตามเวลาที่ปลอดภัยและ retry batch เดิมสูงสุด 3 ครั้ง โดยไม่ย้อนกลับไปเริ่ม FAQ ข้อแรกในรอบเดียวกัน
บน Streamlit Cloud ใช้ Secrets/environment ชื่อเดียวกัน
นำ `faq_embeddings.npz` และ `faq_embeddings.meta.json` ขึ้นพร้อมกันหลังสร้างสำเร็จ ทั้งสองไฟล์ไม่มี API key
metadata ตรวจ SHA-256 ของ FAQ และ binary, embedding model, จำนวน FAQ, format version และตรวจ shape/finite values ของเวกเตอร์
การอ่าน NPZ ใช้ `allow_pickle=False` หากไฟล์ขาด/เสีย/เก่า ใช้ TF-IDF ต่อและแสดงคำแนะนำผู้ดูแลใน expander

## ตรวจสอบก่อนใช้งาน
```powershell
python -m py_compile app.py build_embeddings.py rag.py
python -m unittest discover -s . -p test_rag.py -v
python -m unittest discover -s . -p test_app.py -v
git diff --check
git diff --stat
git status --short
```

การรัน test discovery จะไม่เรียก Gemini จริง เพราะ `test_gemini.py` ทำงานเฉพาะเมื่อสั่งไฟล์นั้นโดยตรง
ชุดทดสอบอัตโนมัติจำลอง Gemini และไม่อ่าน key จริง

## ผลตรวจรอบนี้
- Syntax และ unit tests สำหรับ synonym, A/B/C, parser, metadata, builder batch, fallback/cooldown, history, score normalization และ streaming
- Streamlit AppTest: เริ่มแอปไม่มี API call, Direct ไม่มี API call, generation streaming, stream ขาดกลางทาง, ไม่มี key แล้วยังตอบ Direct
- คำถามต้นฉบับ 150 FAQ: A 124, B 8, C 18; A เลือก FAQ ถูกทุกข้อในชุดนี้
- ตัวอย่าง TF-IDF ในเครื่องใช้ประมาณ 9–15 ms ต่อคำถาม (ไม่ใช่เวลารวม Gemini และไม่ใช่ benchmark บน Cloud)

ตรวจพบไฟล์ embeddings จริงจำนวน 150 รายการ ขนาดเวกเตอร์ 3,072 และ checksum ตรงกับ FAQ แต่รอบนี้ไม่ได้เรียก Gemini จริงเพื่อไม่ใช้โควตา จึงยังไม่ยืนยันโควตา เวลา API หรือความถูกต้องของคำตอบที่โมเดลเรียบเรียงจริง
ระบบย้ายจาก SDK `google-generativeai` ที่เลิกดูแลแล้วไปใช้ `google-genai` และรองรับ `send_message_stream(...)`
ผลทดสอบกับคำถามตรง FAQ ไม่รับประกันคำถามถอดความทุกแบบ ควรทดสอบคำถามจริงก่อนลดเกณฑ์ Direct

แก้เฉพาะ working tree ไม่มี commit/push ก่อนนำขึ้น Git ตรวจรายชื่อไฟล์ให้ไม่มี `.env`, `.streamlit/secrets.toml`, `__pycache__` หรือไฟล์ลับ
