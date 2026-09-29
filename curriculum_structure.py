"""Source-derived curriculum sections and reliable answers to broad course questions."""

import re

from vocabulary import clean_question


BRANCH_NAMES = {
    "power": "วิศวกรรมระบบไฟฟ้ากำลังและระบบควบคุม",
    "electronics": "วิศวกรรมอิเล็กทรอนิกส์และโทรคมนาคม",
}
COURSE = re.compile(r"^(\d{9}|\d{2}x{7})\*?\s+(.+?)\s+(\d+)\(([^)]*)\)")
GROUP = re.compile(r"^[ก-จ]\.\s+(กลุ่มวิชา.+?\s+\d+\s*หน่วยกิต)")
YEAR_WORDS = {"ปีหนึ่ง": 1, "ปีแรก": 1, "ปีสอง": 2, "ปีสาม": 3,
              "ปีสี่": 4, "ปีห้า": 5}


def metadata(record):
    page = record["pdf_page"]
    result = {"section": "อื่น ๆ", "year": None, "term": None, "branch": None}
    if page == 17:
        result["section"] = "โครงสร้างหลักสูตรและหมวดวิชา"
    elif 18 <= page <= 19:
        result["section"] = "หมวดวิชาศึกษาทั่วไปและรายวิชา"
    elif 20 <= page <= 25:
        result["section"] = "หมวดวิชาเฉพาะและรายวิชา"
    elif 26 <= page <= 43:
        result["section"] = "แผนการเรียนตามชั้นปี ภาคเรียน และแขนง"
        result["branch"] = "power" if page <= 34 else "electronics"
        match = re.search(r"ปีที่\s*([1-5])\s*ภาคการศึกษาที่\s*([12])", record["title"])
        if match:
            result["year"], result["term"] = map(int, match.groups())
    elif 44 <= page <= 86:
        result["section"] = "คำอธิบายรายวิชา"
    elif 87 <= page <= 100:
        result["section"] = "อาจารย์และผู้สอนตามเล่มหลักสูตร พ.ศ. 2566"
    elif 101 <= page <= 102:
        result["section"] = "การฝึกประสบการณ์ภาคสนาม"
    elif 103 <= page <= 155:
        result["section"] = "ผลการเรียนรู้และการบริหารหลักสูตร"
    elif 156 <= page <= 200:
        result["section"] = "ภาคผนวกหลักสูตร"
    elif page >= 201:
        result["section"] = "ประวัติและการเปรียบเทียบหลักสูตรเดิม"
    return result


def understand(question):
    text = clean_question(question).casefold()
    years = [(match.start(), int(match.group(1))) for match in
             re.finditer(r"(?:ชั้น)?ปี(?:ที่)?\s*([1-5])(?!\d)", text)]
    years.extend((match.start(), number) for word, number in YEAR_WORDS.items()
                 for match in re.finditer(word, text))
    year = max(years)[1] if years else None
    terms = list(re.finditer(r"(?:เทอม|ภาค(?:เรียน|การศึกษา)?)(?:ที่)?\s*([12])", text))
    term = int(terms[-1].group(1)) if terms else None
    branch_mentions = [(match.start(), "power") for match in
                       re.finditer(r"ไฟฟ้ากำลัง|ระบบควบคุม|สายกำลัง", text)]
    branch_mentions.extend((match.start(), "electronics") for match in
                           re.finditer(r"อิเล็กทรอนิกส์|โทรคมนาคม|(?:สาย|แขนง(?:วิชา)?)สื่อสาร", text))
    branch = max(branch_mentions)[1] if branch_mentions else None
    general = bool(re.search(r"ศึกษาทั่วไป|วิชาทั่วไป|หมวดทั่วไป|gen\s*ed", text))
    language = bool(re.search(r"ภาษาอังกฤษ|กลุ่มวิชาภาษา", text))
    subject = bool(re.search(r"รายวิชา|วิชา|เรียน|สอน", text))
    broad = (not year and not branch and not general and not re.search(r"\d{9}|PLC|ไมโคร|ฝึก|ค่า|สมัคร", text, re.I)
             and len(text) <= 35 and subject
             and bool(re.search(r"อะไรบ้าง|ไรบ้าง|วิชา(?:อะไร|ไร|ไหน)|เรียน(?:อะไร|ไร)|มีวิชา", text)))
    kind = ("general_plan" if year and general else "plan" if year and subject else
            "language" if language and subject and not year else "general" if general else
            "branch" if branch and subject else "overview" if broad else "other")
    return {"kind": kind, "year": year, "term": term, "branch": branch}


def course_rows(record):
    rows = []
    for line in record["body"].splitlines():
        match = COURSE.match(line.strip())
        if match:
            rows.append({"code": match.group(1), "name": match.group(2).strip(),
                         "credits": int(match.group(3))})
    return rows


def source(record):
    return f"หน้า PDF {record['pdf_page']} (หน้าในเล่ม {record['printed_page']})"


def overview_answer(records):
    structure = next(record for record in records if record["pdf_page"] == 17)
    text = structure["body"]
    total = re.search(r"รวมตลอดหลักสูตร\s*(\d+)\s*หน่วยกิต", text)
    groups = re.findall(r"(?:หมวดวิชาศึกษาทั่วไป|หมวดวิชาเฉพาะ|หมวดวิชาเลือกเสรี)\s*(\d+)\s*หน่วยกิต", text)
    if not total or len(groups) < 3:
        return None
    general = next(record for record in records if record["pdf_page"] == 18)
    education = next(record for record in records if record["pdf_page"] == 20)
    examples = [row["name"] for row in course_rows(general)[:3]]
    teacher = [row["name"] for row in course_rows(education)[:3]]
    return (f"หลักสูตรนี้เรียนรวม {total.group(1)} หน่วยกิต แบ่งเป็นวิชาศึกษาทั่วไป {groups[0]} หน่วยกิต "
            f"วิชาเฉพาะ {groups[1]} หน่วยกิต และวิชาเลือกเสรี {groups[2]} หน่วยกิต\n\n"
            f"- **วิชาศึกษาทั่วไป:** เช่น {', '.join(examples)}\n"
            f"- **วิชาเฉพาะ:** มีวิชาด้านการศึกษา คณิตศาสตร์/วิทยาศาสตร์ และวิศวกรรมไฟฟ้า "
            f"ตัวอย่างวิชาด้านการศึกษา ได้แก่ {', '.join(teacher)} และมีวิชาตามแขนงที่เลือก\n"
            f"- **วิชาเลือกเสรี:** เลือกตามเงื่อนไขในเล่มหลักสูตร\n\n"
            f"รายวิชาที่เรียนในแต่ละภาคเรียนดูได้จากแผนการศึกษาของแต่ละแขนง "
            f"หากบอกชั้นปีหรือแขนง จะระบุรายวิชาได้ละเอียดขึ้น\n\n"
            f"แหล่งข้อมูล: {source(structure)}, {source(general)}, {source(education)}")


def general_answer(records):
    pages = [record for record in records if record["pdf_page"] in (18, 19)]
    if len(pages) != 2:
        return None
    groups = []
    current = None
    for record in pages:
        for line in record["body"].splitlines():
            line = line.strip()
            heading = GROUP.match(line)
            if heading:
                current = {"name": heading.group(1), "courses": []}
                groups.append(current)
            course = COURSE.match(line)
            if course and current is not None:
                current["courses"].append(course.group(2).strip())
    if len(groups) != 5:
        return None
    parts = ["หมวดวิชาศึกษาทั่วไปมี 30 หน่วยกิต แบ่งเป็น 5 กลุ่มตามเล่มหลักสูตร:"]
    for group in groups:
        names = group["courses"]
        shown = ", ".join(names[:4])
        if len(names) > 4:
            shown += " และวิชาอื่นในกลุ่ม"
        parts.append(f"- **{group['name']}** — วิชาที่ระบุในเล่ม เช่น {shown}")
    parts.append("รายวิชาที่เป็นตัวเลือกไม่จำเป็นต้องเรียนทุกวิชาที่แสดง ให้ดูเงื่อนไขของแต่ละกลุ่มและแผนตามรุ่น")
    parts.append(f"แหล่งข้อมูล: {source(pages[0])}, {source(pages[1])}")
    return "\n".join(parts)


def language_answer(records):
    record = next(record for record in records if record["pdf_page"] == 18)
    text = record["body"].split("ก. กลุ่มวิชาภาษา", 1)[1].split("ข. กลุ่มวิชาบูรณาการ", 1)[0]
    compulsory, choices = text.split("- วิชาเลือก 6 หน่วยกิต", 1)
    required = course_rows({"body": compulsory})
    optional = course_rows({"body": choices})
    if not required or not optional:
        return None
    count_note = ("วิชาบังคับมี 2 วิชา วิชาเลือกตัวอย่างในเล่มมีวิชาละ 3 หน่วยกิต "
                  "จึงเลือกให้ครบอีก 6 หน่วยกิตตามเงื่อนไข\n\n"
                  if len(required) == 2 and all(row["credits"] == 3 for row in required + optional)
                  else "")
    return ("กลุ่มวิชาภาษาในหมวดศึกษาทั่วไปมี 12 หน่วยกิต: วิชาบังคับ 6 หน่วยกิต "
            "และวิชาเลือก 6 หน่วยกิต\n\n" + count_note +
            f"- **วิชาบังคับ:** {', '.join(row['name'] for row in required)}\n"
            f"- **ตัวอย่างวิชาเลือก:** {', '.join(row['name'] for row in optional)} "
            "หรือรายวิชาอื่นในกลุ่มที่เปิดสอนและภาควิชาเห็นชอบ\n\n"
            f"แหล่งข้อมูล: {source(record)}")


def branch_answer(records, intent):
    if intent["branch"] == "power":
        first = next(record for record in records if record["pdf_page"] == 23)
        second = next(record for record in records if record["pdf_page"] == 24)
        mandatory_text, optional_text = first["body"].split("- วิชาเลือก 3 หน่วยกิต", 1)
        optional_text += second["body"].split("- แขนงวิชาวิศวกรรมอิเล็กทรอนิกส์", 1)[0]
    else:
        first = next(record for record in records if record["pdf_page"] == 24)
        second = next(record for record in records if record["pdf_page"] == 25)
        mandatory_text = first["body"].split("- แขนงวิชาวิศวกรรมอิเล็กทรอนิกส์", 1)[1]
        optional_text = second["body"]
    mandatory = course_rows({"body": mandatory_text})
    optional = course_rows({"body": optional_text})
    if not mandatory or not optional:
        return None
    names = ", ".join(row["name"] for row in mandatory)
    options = ", ".join(row["name"] for row in optional[:4])
    return (f"แขนง{BRANCH_NAMES[intent['branch']]}มีวิชาเฉพาะแขนง 33 หน่วยกิต "
            f"แบ่งเป็นวิชาบังคับ 30 หน่วยกิต และวิชาเลือก 3 หน่วยกิต\n\n"
            f"- **วิชาบังคับที่ระบุในเล่ม:** {names}\n"
            f"- **ตัวอย่างวิชาเลือก:** {options} (เลือกตามเงื่อนไข ไม่ได้เรียนทุกวิชา)\n\n"
            f"แหล่งข้อมูล: {source(first)}, {source(second)}")


def plan_records(records, intent):
    result = [record for record in records
              if record.get("section") == "แผนการเรียนตามชั้นปี ภาคเรียน และแขนง"
              and record.get("year") == intent["year"]
              and (intent["term"] is None or record.get("term") == intent["term"])
              and (intent["branch"] is None or record.get("branch") == intent["branch"])]
    return sorted(result, key=lambda item: (item["branch"] != "power", item["term"]))


def plan_answer(records, intent):
    plans = plan_records(records, intent)
    if not plans:
        return None
    branches = {record["branch"] for record in plans}
    # Identical first-year plans need only one list, with both sources shown.
    if len(branches) == 2:
        power = [record for record in plans if record["branch"] == "power"]
        electronics = [record for record in plans if record["branch"] == "electronics"]
        same = len(power) == len(electronics) and all(
            sorted((row["code"], row["name"], row["credits"]) for row in course_rows(a)) ==
            sorted((row["code"], row["name"], row["credits"]) for row in course_rows(b))
            for a, b in zip(power, electronics))
        if same:
            shown = power
            heading = f"แผนชั้นปีที่ {intent['year']} ของทั้งสองแขนงมีรายวิชาที่ระบุตรงกันในภาคเรียนที่แสดง:"
        else:
            shown = plans
            heading = f"แผนชั้นปีที่ {intent['year']} แยกตามแขนงดังนี้:"
    else:
        shown = plans
        branch = BRANCH_NAMES[plans[0]["branch"]]
        heading = f"แผนชั้นปีที่ {intent['year']} แขนง{branch}:"
    parts = [heading]
    for record in shown:
        rows = course_rows(record)
        if not rows:
            return None
        label = f"**ภาคเรียนที่ {record['term']}"
        if len(branches) == 2 and shown is plans:
            label += f" · {BRANCH_NAMES[record['branch']]}"
        label += "**"
        courses = ", ".join(f"{row['name']} ({row['credits']} หน่วยกิต)" for row in rows)
        parts.append(f"- {label}: {courses} — {source(record)}")
    if len(shown) != len(plans):
        parts.append("ข้อมูลแขนงที่สองตรงกับรายวิชาข้างต้น: " + ", ".join(source(record) for record in electronics))
    parts.append("รายการอาจมีวิชาเลือกตามเงื่อนไขของแต่ละแขนง ให้ตรวจแผนตามรุ่นก่อนลงทะเบียน")
    return "\n".join(parts)


def general_plan_answer(records, intent):
    general_pages = [record for record in records if record["pdf_page"] in (18, 19)]
    general_codes = {row["code"] for record in general_pages for row in course_rows(record)}
    plans = plan_records(records, intent)
    selected = []
    for record in plans:
        rows = [row for row in course_rows(record)
                if row["code"] in general_codes or row["code"] == "08xxxxxxx"]
        if rows:
            selected.append((record, rows))
    if not selected:
        return None
    parts = [f"วิชาศึกษาทั่วไปที่ปรากฏในแผนชั้นปีที่ {intent['year']} มีดังนี้:"]
    grouped = {}
    for record, rows in selected:
        key = (record["term"], tuple((row["code"], row["name"]) for row in rows))
        grouped.setdefault(key, {"rows": rows, "records": []})["records"].append(record)
    for item in grouped.values():
        rows = item["rows"]
        citations = item["records"]
        record = citations[0]
        label = f"ภาคเรียนที่ {record['term']}"
        if len(citations) > 1:
            label += " · ทั้งสองแขนง"
        elif intent["branch"] is None and len({item["branch"] for item, _ in selected}) > 1:
            label += f" · แขนง{BRANCH_NAMES[record['branch']]}"
        names = ", ".join(f"{row['name']} ({row['credits']} หน่วยกิต)" for row in rows)
        parts.append(f"- **{label}:** {names} — {', '.join(source(item) for item in citations)}")
    parts.append("วิชาเลือกในหมวดศึกษาทั่วไปอาจเลือกจากรายวิชาอื่นที่มหาวิทยาลัยเปิดสอนตามเงื่อนไขในเล่ม")
    parts.append("โครงสร้างหมวด: " + ", ".join(source(record) for record in general_pages))
    return "\n".join(parts)


def structured_answer(records, intent):
    if intent["kind"] == "overview":
        return overview_answer(records)
    if intent["kind"] == "general":
        return general_answer(records)
    if intent["kind"] == "language":
        return language_answer(records)
    if intent["kind"] == "branch":
        return branch_answer(records, intent)
    if intent["kind"] == "plan":
        return plan_answer(records, intent)
    if intent["kind"] == "general_plan":
        return general_plan_answer(records, intent)
    return None
