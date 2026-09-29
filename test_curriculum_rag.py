"""Source integrity and retrieval behavior for the 2566 curriculum."""

from pathlib import Path
import unittest
from unittest.mock import Mock

from curriculum_rag import (DATASET_NAME, allowed_records, build_retriever,
                            fallback_answer, load_embeddings, parse_curriculum, retrieve)
from rag import CONTEXT_STATE_KEY, NOT_FOUND


ROOT = Path(__file__).resolve().parent


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / DATASET_NAME).read_bytes()
        cls.retriever = build_retriever(cls.source, ROOT)

    def test_every_pdf_content_page_has_a_sourced_chunk(self):
        records = parse_curriculum(self.source)

        pages = {record["pdf_page"] for record in records}
        self.assertEqual(pages, set(range(5, max(pages) + 1)))
        self.assertTrue(len(records) >= len(pages))
        self.assertTrue(all(record["body"] and record["printed_page"] == record["pdf_page"] - 4
                            for record in records))


    def test_overview_and_course_code_find_correct_pages(self):
        for question, page in (("หลักสูตรนี้เรียนกี่ปี", 5),
                               ("ต้องเรียนทั้งหมดกี่หน่วยกิต", 5),
                               ("วิชา 020253004 คืออะไร", 56)):
            result = retrieve(question, self.retriever, {}, None)
            index = result["matches"][0][0]
            self.assertEqual(self.retriever["records"][index]["pdf_page"], page)
            self.assertIn("หน้า PDF", result["context"])

    def test_old_comparison_and_ocr_are_filtered_by_default(self):
        records = self.retriever["records"]
        standard = allowed_records("วิชานี้เรียนอะไร", records)
        self.assertFalse(any(standard[index] for index, item in enumerate(records)
                             if (201 <= item["pdf_page"] <= 230) or item["ocr"]))
        historical = allowed_records("เปรียบเทียบหลักสูตร 2561", records)
        self.assertTrue(any(historical[index] for index, item in enumerate(records)
                            if (201 <= item["pdf_page"] <= 230)))


    def test_current_fee_does_not_come_from_2023_document(self):
        result = retrieve("ค่าเทอมล่าสุดปี 2569", self.retriever, {}, None)
        self.assertEqual(result["answer"], NOT_FOUND)
        self.assertFalse(result["matches"])

    def test_context_and_stale_embeddings(self):
        first = retrieve("หลักสูตรนี้มีกี่แขนงวิชาให้เลือก", self.retriever, {}, None)
        next_state = {CONTEXT_STATE_KEY: first["next_context"]}
        follow = retrieve("อะไรบ้าง", self.retriever, next_state, None)
        self.assertTrue(follow["used_context"])
        self.assertIn("แขนงวิชา", follow["search_question"])
        stale, warning = load_embeddings(ROOT, self.source + b"\nchanged", 253)
        self.assertIsNone(stale)
        self.assertTrue(warning)

    def test_offline_excerpt_uses_the_relevant_passage(self):
        result = retrieve("วิชา 020253004 คืออะไร", self.retriever, {}, None)
        answer = fallback_answer(result, self.retriever)
        self.assertIn("020253004", answer)
        self.assertIn("หน้า PDF 56", answer)

    def test_natural_overview_and_general_education_are_complete_without_api(self):
        embedding = Mock(side_effect=AssertionError("should not request embedding"))
        overview = retrieve("เรียนวิชาอะไรบ้าง", self.retriever, {}, embedding)
        self.assertEqual(overview["route"], "A")
        self.assertIn("175 หน่วยกิต", overview["answer"])
        self.assertIn("วิชาด้านการศึกษา", overview["answer"])
        self.assertEqual([self.retriever["records"][i]["pdf_page"] for i, _ in overview["matches"]],
                         [17, 18, 20])
        general = retrieve("วิชาศึกษาทั่วไปมีอะไรบ้าง", self.retriever, {}, embedding)
        self.assertIn("5 กลุ่ม", general["answer"])
        self.assertIn("ภาษาอังกฤษ 2", general["answer"])
        self.assertIn("กีฬาและนันทนาการ", general["answer"])
        self.assertNotEqual(overview["answer"], general["answer"])
        embedding.assert_not_called()

    def test_year_plan_handles_colloquial_questions_without_duplicates(self):
        first = retrieve("ปีหนึ่งเรียนอะไรบ้าง", self.retriever, {}, None)
        self.assertEqual(first["route"], "A")
        self.assertIn("ของทั้งสองแขนง", first["answer"])
        self.assertEqual(first["answer"].count("หลักวิชาชีพครู (3 หน่วยกิต)"), 1)
        self.assertIn("หน้า PDF 26", first["answer"])
        self.assertIn("วิชาเลือกในกลุ่มวิชากีฬาและนันทนาการ", first["answer"])
        second = retrieve("ปี 2 สายไฟฟ้ากำลังเรียนอะไร", self.retriever, {}, None)
        self.assertEqual([self.retriever["records"][i]["pdf_page"] for i, _ in second["matches"]],
                         [28, 29])
        self.assertIn("วงจรไฟฟ้า", second["answer"])
        communication = retrieve("ปี 3 แขนงสื่อสารเรียนอะไร", self.retriever, {}, None)
        self.assertEqual([self.retriever["records"][i]["pdf_page"]
                          for i, _ in communication["matches"]], [39, 40])

    def test_follow_up_changes_year_or_branch_without_losing_the_topic(self):
        first = retrieve("ปี 1 สายไฟฟ้ากำลังเรียนอะไร", self.retriever, {}, None)
        state = {CONTEXT_STATE_KEY: first["next_context"]}
        second = retrieve("แล้วปี 2 ล่ะ", self.retriever, state, None)
        self.assertTrue(second["used_context"])
        self.assertEqual([self.retriever["records"][i]["pdf_page"]
                          for i, _ in second["matches"]], [28, 29])
        branch = retrieve("แล้วแขนงสื่อสารล่ะ", self.retriever, state, None)
        self.assertTrue(branch["used_context"])
        self.assertEqual([self.retriever["records"][i]["pdf_page"]
                          for i, _ in branch["matches"]], [35, 36])

    def test_broad_fallback_never_presents_one_course_as_full_list(self):
        result = retrieve("ภาควิชานี้สอนวิชาอะไร", self.retriever, {}, None)
        answer = fallback_answer(result, self.retriever)
        self.assertIn("ยังไม่ควรถือเป็นคำตอบครบถ้วน", answer)
        self.assertNotIn("080103001", answer)

    def test_branch_courses_and_plc_evidence(self):
        power = retrieve("เลือกสายไฟฟ้ากำลังเรียนอะไร", self.retriever, {}, None)
        self.assertIn("วิชาบังคับ 30 หน่วยกิต", power["answer"])
        self.assertIn("การออกแบบระบบไฟฟ้า", power["answer"])
        self.assertEqual([self.retriever["records"][i]["pdf_page"]
                          for i, _ in power["matches"]], [23, 24])
        electronics = retrieve("แขนงอิเล็กทรอนิกส์เรียนวิชาไรบ้าง", self.retriever, {}, None)
        self.assertIn("ระบบการสื่อสารไร้สาย", electronics["answer"])
        plc = retrieve("มี PLC ไหม", self.retriever, {}, None)
        evidence = fallback_answer(plc, self.retriever)
        self.assertIn("ระบบควบคุมอัตโนมัติสมัยใหม่", evidence)
        self.assertIn("หน้า PDF 64", evidence)

    def test_general_courses_for_a_specific_year_are_filtered(self):
        result = retrieve("วิชาศึกษาทั่วไปปี 1 มีอะไรบ้าง", self.retriever, {}, None)
        self.assertEqual(result["route"], "A")
        self.assertIn("ภาษาอังกฤษ 1", result["answer"])
        self.assertIn("กีฬาและนันทนาการ", result["answer"])
        self.assertIn("ทั้งสองแขนง", result["answer"])
        self.assertNotIn("การเขียนแบบวิศวกรรม", result["answer"])

    def test_english_group_distinguishes_required_from_choices(self):
        result = retrieve("ต้องเรียนภาษาอังกฤษตัวไหนบ้าง", self.retriever, {}, None)
        self.assertEqual(result["route"], "A")
        self.assertIn("วิชาบังคับมี 2 วิชา", result["answer"])
        self.assertIn("ภาษาอังกฤษ 1, ภาษาอังกฤษ 2", result["answer"])
        self.assertIn("ตัวอย่างวิชาเลือก", result["answer"])
        self.assertIn("หน้า PDF 18", result["answer"])


if __name__ == "__main__":
    unittest.main()
