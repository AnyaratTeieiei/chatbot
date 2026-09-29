"""Domain regression cases; all checks run offline without real API requests."""
import re
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import rag
from vocabulary import normalize_query

ROOT = Path(__file__).resolve().parent
CASES = [
    ('เบอร์โทรครุไฟฟ้า', '178'),
    ('เบอร์ติดต่อครุไฟฟ้า', '178'),
    ('อีเมลภาควิชา', '178'),
    ('ครุไฟฟ้าอยู่คณะอะไร', '172'),
    ('ครุอุตมีสาขาอะไรบ้าง', '180'),
    ('FTE มีภาคอะไรบ้าง', '180'),
    ('ครุไฟฟ้ามีหลักสูตรอะไรบ้าง', '173'),
    ('คอบกับวศบต่างกันยังไง', '173'),
    ('ปวสเรียนต่อกี่ปี', '174'),
    ('ปวสต่อครุไฟฟ้า', '174'),
    ('TEE เรียนกี่ปี', '175'),
    ('tee คืออะไร', '175'),
    ('หัวหน้าภาคคือใคร', '176'),
    ('หัวหน้าภาควิชาเทคโนโลยีวิศวกรรมไฟฟ้าคือใคร', '114'),
    ('ใครเป็นผู้บริหารสูงสุดของภาควิชาเทคโนโลยีวิศวกรรมไฟฟ้า', '150'),
    ('ภาคนี้มีอาจารย์กี่คน', '177'),
    ('รายชื่อคณาจารย์', '125'),
    ('ที่อยู่ครุไฟฟ้า', '179'),
    ('ปีหนึ่งเรียนอะไร', '076'),
    ('ขอแผนการเรียนปี 1', '183'),
    ('ค่าเทอมปีนี้', '182'),
    ('ค่าเทอมเทียบโอน', '182'),
    ('ค่าเทิมล่าสุดของครุไฟฟ้าเท่าไหร่', '182'),
    ('กำหนดการรับสมัครล่าสุด', '181'),
    ('เปิดรับสมัครเมื่อไหร่', '181'),
    ('เปิดรับสมัครวันไหน', '181'),
    ('คะแนนขั้นต่ำปีนี้', '181'),
    ('ติดต่อ TCAS ที่ไหน', '040'),
    ('ทุนการศึกษามีไหม', '106'),
    ('กยศ กู้ได้ไหม', '106'),
    ('กู้เรียนได้มั้ย', '106'),
    ('สามารถขอทุนได้ไหม?', '106'),
    ('กู้ กยศ. ได้ไหม?', '106'),
    ('เลือกแทร็กตอนไหน', '023'),
    ('มีห้องแลปอะไรบ้าง', '167'),
    ('เรียนจบได้ตั๋วครูเลยไหม', '014'),
]


class DomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = rag.build_retriever(
            (ROOT / 'FAQ_Chatbot_100.md').read_bytes(), ROOT)

    def test_paraphrases_rank_correct_source(self):
        for question, expected in CASES:
            with self.subTest(question=question):
                _, indexes = rag.search_tfidf(question, self.retriever)
                self.assertEqual(rag.faq_reference(self.retriever['chunks'][indexes[0]]), 'FAQ ' + expected)

    def test_all_canonical_questions_still_rank_themselves(self):
        for index, chunk in enumerate(self.retriever['chunks']):
            question = re.search(r'\*\*คำถาม:\*\*\s*(.+)', chunk).group(1)
            with self.subTest(question=question):
                _, indexes = rag.search_tfidf(question, self.retriever)
                self.assertEqual(indexes[0], index)

    def test_every_alias_has_consistent_canonical_meaning(self):
        for canonical, aliases in rag.SYNONYM_GROUPS.items():
            for alias in aliases:
                with self.subTest(alias=alias):
                    self.assertEqual(normalize_query(alias), canonical)

    def test_not_equivalent_concepts_stay_separate(self):
        for left, right in [('ทุนการศึกษา', 'กยศ'), ('ฝึกงาน', 'ฝึกสอน'),
                            ('ฝึกงาน', 'สหกิจ'), ('GPA', 'GPAX'),
                            ('ค.อ.บ.', 'วศ.บ.'), ('ปวช', 'ปวส'),
                            ('ตั๋วครู', 'ใบ กว.'), ('ดรอปวิชา', 'ดรอปเทอม')]:
            self.assertNotEqual(normalize_query(left), normalize_query(right))
        self.assertEqual(normalize_query('collaborate laboratory'), 'collaborate ห้องปฏิบัติการ')
        self.assertEqual(normalize_query('คณะนี้ สาขา ภาควิชา'), 'คณะนี้ สาขา ภาควิชา')
        self.assertEqual(normalize_query('steam'), 'steam')
        self.assertNotEqual(normalize_query('มัธยมปลาย'), normalize_query('ม.6'))
        self.assertIn('ไฟฟ้า', normalize_query('ปวส.ไฟฟ้า'))
        self.assertNotIn('ภาควิชาครุศาสตร์ไฟฟ้า', normalize_query('ภาควิชาเทคโนโลยีวิศวกรรมไฟฟ้า'))

    def test_entity_name_does_not_hide_conversation_topic(self):
        self.assertEqual(rag.detect_topic('ค่าเทอมมหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ'), 'ค่าเทอม')

    def test_verified_fast_answers_need_no_api(self):
        embed = Mock(side_effect=AssertionError('API must not be called'))
        for question in ['เบอร์โทรครุไฟฟ้า', 'ปวสเรียนต่อกี่ปี', 'ภาคนี้มีอาจารย์กี่คน', 'หัวหน้าภาคคือใคร']:
            result = rag.retrieve(question, self.retriever, {}, embed)
            self.assertEqual(result['route'], 'A')
            self.assertIn('https://te.kmutnb.ac.th/', result['answer'])
            if question == 'ภาคนี้มีอาจารย์กี่คน':
                self.assertIn('20 คน', result['answer'])
        embed.assert_not_called()

    def test_department_head_answers_are_consistent(self):
        questions = [
            'หัวหน้าภาคคือใคร',
            'หัวหน้าภาควิชาเทคโนโลยีวิศวกรรมไฟฟ้าคือใคร',
            'ใครเป็นผู้บริหารสูงสุดของภาควิชาเทคโนโลยีวิศวกรรมไฟฟ้า',
        ]
        for question in questions:
            answer = rag.retrieve(question, self.retriever, {}, None)['answer']
            self.assertIn('ภานี น้อยยิ่ง', answer)
            self.assertNotIn('สุริโยทัย สุปัญญาพงศ์ เป็นหัวหน้า', answer)

    def test_archived_faq_file_retains_its_records(self):
        self.assertEqual(len(self.retriever['chunks']), 184)
        self.assertTrue(any(rag.faq_reference(c) == 'FAQ 102' for c in self.retriever['chunks']))
        self.assertTrue(any(rag.faq_reference(c) == 'FAQ 183' for c in self.retriever['chunks']))

    def test_question_aliases_cannot_reference_missing_faq(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / 'question_aliases.json').write_text(
                '{"FAQ 999": ["คำถามที่ไม่มีปลายทาง"]}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'FAQ 999'):
                rag.build_retriever((ROOT / 'FAQ_Chatbot_100.md').read_bytes(), target)

    def test_hybrid_scores_cover_all_faqs(self):
        dimensions = 3
        retriever = dict(self.retriever, semantic=np.ones((len(self.retriever['chunks']), dimensions)))
        embed = Mock(return_value={'embedding': [1.] * dimensions})
        result = rag.retrieve('อธิบายความต่างของหลักสูตรและวิธีสมัคร', retriever, {}, embed)
        embed.assert_called_once()
        self.assertEqual(result['route'], 'C')
        self.assertTrue(all(np.isfinite(s) for _, s in result['matches']))

    def test_context_switch_keeps_loan_and_scholarship_separate(self):
        state = {rag.CONTEXT_STATE_KEY: {'topic': 'ทุนการศึกษา',
                 'source_question': 'ทุนการศึกษามีไหม', 'faq_indexes': [84]}}
        result = rag.retrieve('แล้วกยศ กู้ได้ไหม', self.retriever, state, None)
        self.assertFalse(result['used_context'])


if __name__ == '__main__':
    unittest.main()
