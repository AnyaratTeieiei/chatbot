import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import rag
from build_embeddings import (EMBEDDING_WINDOW_DELAY_SECONDS, build_embeddings,
                              rate_limit_retry_delay)

ROOT = Path(__file__).resolve().parent


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = (ROOT / 'FAQ_Chatbot_100.md').read_bytes()
        cls.retriever = rag.build_retriever(cls.data, ROOT)

    def test_synonyms(self):
        for question, expected in [('ครูประจำภาควิชามีกี่คน', 'ผู้สอน'),
                                   ('ค่าเล่าเรียนเท่าไหร่', 'ค่าเทอม'),
                                   ('ปีหนึ่งเรียนอะไร', 'ปี 1')]:
            self.assertIn(expected, rag.expand_query(question))
        self.assertIn('อาจารย์ประจำวิชา', rag.SYNONYM_GROUPS['ผู้สอน'])

    def test_routes(self):
        for scores, route in [([.9, .3], 'A'), ([.45, .25], 'B'),
                              ([.1, .01], 'C'), ([.8, .79], 'C'), ([], 'C')]:
            self.assertEqual(rag.choose_route(scores), route)
        self.assertEqual(rag.choose_route([.95, .1], 'อธิบายค่าเทอมและหน่วยกิต'), 'C')

    def test_answer_parser(self):
        self.assertEqual(rag.extract_faq_answer('## FAQ 001\n**คำถาม:** Q\n**คำตอบ:** A\n- B'), 'A\n- B')
        self.assertIsNone(rag.extract_faq_answer('unknown'))
        self.assertTrue(all(rag.extract_faq_answer(c) for c in self.retriever['chunks']))

    def test_generation_fallback_requires_clear_local_match(self):
        result = rag.retrieve('เปิดรับสมัครเมื่อไหร่', self.retriever, {}, None)
        answer = rag.fallback_faq_answer(result, self.retriever)
        self.assertIn('FAQ 181', answer)
        self.assertIn('Admission KMUTNB', answer)
        uncertain = dict(result, matches=[(result['matches'][0][0], .3), (result['matches'][1][0], .29)])
        self.assertIsNone(rag.fallback_faq_answer(uncertain, self.retriever))

    def test_faq_parser_rejects_duplicate_or_incomplete_records(self):
        with self.assertRaisesRegex(ValueError, 'FAQ 001'):
            rag.parse_faq('## FAQ 001\n**คำถาม:** Q\n**คำตอบ:** A\n\n'
                          '## FAQ 001\n**คำถาม:** Q2\n**คำตอบ:** A2')
        with self.assertRaisesRegex(ValueError, 'ไม่มีคำตอบ'):
            rag.parse_faq('## FAQ 001\n**คำถาม:** Q')

    def test_local_retrieval(self):
        for question in ('ครูประจำภาควิชามีกี่คน', 'ค่าเล่าเรียนเท่าไหร่', 'ปีหนึ่งเรียนอะไร'):
            scores, indexes = rag.search_tfidf(question, self.retriever)
            self.assertEqual(len(scores), len(self.retriever['chunks']))
            self.assertEqual(len(indexes), 3)
            self.assertTrue(np.isfinite(scores).all())

    def test_high_confidence_teacher_count_is_answered_directly(self):
        result = rag.retrieve('ภาควิชามีคณาจารย์ประจำกี่คน?', self.retriever, {}, None)
        self.assertEqual(result['route'], 'A')
        self.assertIn('20 คน', result['answer'])
        self.assertIn('FAQ 124', result['answer'])
        self.assertEqual(result['matches'][0][0], 123)

    def test_contextual_follow_up_reuses_last_topic_and_faqs(self):
        state = {
            rag.CONTEXT_STATE_KEY: {
                'topic': 'แขนงวิชา',
                'source_question': 'หมายถึงจำนวนแขนงวิชาที่มีให้เลือกในหลัก',
                'faq_indexes': [19, 20, 21],
            }
        }
        result = rag.retrieve('อะไรบ้าง', self.retriever, state, None)
        self.assertTrue(result['used_context'])
        self.assertEqual(result['next_context']['topic'], 'แขนงวิชา')
        self.assertIn('จำนวนแขนงวิชา', result['search_question'])
        self.assertEqual({index for index, _ in result['matches']}, {19, 20, 21})

    def test_contextual_semantic_search_embeds_the_complete_query_once(self):
        state = {
            rag.CONTEXT_STATE_KEY: {
                'topic': 'แขนงวิชา',
                'source_question': 'หมายถึงจำนวนแขนงวิชาที่มีให้เลือกในหลัก',
                'faq_indexes': [19, 20, 21],
            }
        }
        retriever = dict(
            self.retriever,
            semantic=np.ones((len(self.retriever['chunks']), 3)),
        )
        embed = Mock(return_value={'embedding': [1., 1., 1.]})
        with patch('rag.choose_route', return_value='C'):
            result = rag.retrieve('แล้วต้องเลือกตอนไหน', retriever, state, embed)
        self.assertTrue(result['used_context'])
        embed.assert_called_once()
        self.assertIn('แขนงวิชา', embed.call_args.kwargs['content'])
        self.assertIn('แล้วต้องเลือกตอนไหน', embed.call_args.kwargs['content'])

    def test_clear_new_topic_does_not_reuse_last_context(self):
        state = {
            rag.CONTEXT_STATE_KEY: {
                'topic': 'แขนงวิชา',
                'source_question': 'หลักสูตรนี้มีแขนงวิชาอะไรบ้าง',
                'faq_indexes': [19, 20, 21],
            }
        }
        result = rag.retrieve('ค่าเทอมเท่าไหร่', self.retriever, state, None)
        self.assertFalse(result['used_context'])
        self.assertEqual(result['search_question'], 'ค่าเทอมเท่าไหร่')
        self.assertEqual(result['next_context']['topic'], 'ค่าเทอม')

    def test_overlapping_teacher_topics_share_one_canonical_topic(self):
        self.assertEqual(rag.detect_topic('คณาอาจารย์มีใครบ้าง'), 'ผู้สอน')

    def test_follow_up_can_discover_new_faq_in_same_topic(self):
        state = {
            rag.CONTEXT_STATE_KEY: {
                'topic': 'แขนงวิชา',
                'source_question': 'หลักสูตรนี้มีแขนงวิชาอะไรบ้าง',
                'faq_indexes': [19, 20, 21],
            }
        }
        result = rag.retrieve('แล้วต้องเลือกตอนไหน', self.retriever, state, None)
        self.assertTrue(result['used_context'])
        self.assertIn(22, [index for index, _ in result['matches']])

    def test_direct_zero_calls_and_bad_format_fallback(self):
        callback = Mock(side_effect=AssertionError('No API call expected'))
        with patch('rag.search_tfidf', return_value=(np.array([.95, .1]), np.array([0, 1]))):
            result = rag.retrieve('ชื่อหลักสูตร', self.retriever, {}, callback)
            self.assertEqual(result['route'], 'A')
            self.assertIn('FAQ 001', result['answer'])
            self.assertEqual(len(result['matches']), 1)
            with patch('rag.extract_faq_answer', return_value=None):
                self.assertEqual(rag.retrieve('ชื่อหลักสูตร', self.retriever, {}, callback)['route'], 'B')
        callback.assert_not_called()

    def test_embedding_builder_loader_and_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'FAQ_Chatbot_100.md').write_bytes(self.data)
            embed = Mock(side_effect=lambda **kw: {'embedding': [[.1, .2, .3]] * len(kw['content'])})
            sleep = Mock()
            count = build_embeddings(root, embed, sleep_fn=sleep, progress=Mock())
            self.assertEqual(embed.call_count, (count + 19) // 20)
            self.assertEqual(sleep.call_count, (count - 1) // 80)
            self.assertTrue(all(call.args == (EMBEDDING_WINDOW_DELAY_SECONDS,)
                                for call in sleep.call_args_list))
            vectors, warning = rag.load_persisted_embeddings(root, self.data, count)
            self.assertEqual(vectors.shape, (count, 3))
            self.assertIsNone(warning)
            unix_newlines = self.data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
            vectors, warning = rag.load_persisted_embeddings(root, unix_newlines, count)
            self.assertEqual(vectors.shape, (count, 3))
            self.assertIsNone(warning)
            self.assertIsNone(rag.load_persisted_embeddings(root, self.data + b'changed', count)[0])
            meta_path = root / 'faq_embeddings.meta.json'
            original = meta_path.read_text()
            for key, value in [('model', 'wrong'), ('faq_count', 1), ('format_version', -1)]:
                metadata = json.loads(original)
                metadata[key] = value
                meta_path.write_text(json.dumps(metadata))
                self.assertIsNone(rag.load_persisted_embeddings(root, self.data, count)[0])
            meta_path.write_text(original)
            (root / 'faq_embeddings.npz').write_bytes(b'broken')
            self.assertIsNone(rag.load_persisted_embeddings(root, self.data, count)[0])

    def test_embedding_builder_retries_same_batch_after_per_minute_quota(self):
        class ResourceExhausted(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'FAQ_Chatbot_100.md').write_bytes(self.data)
            call_count = 0

            def embed(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ResourceExhausted(
                        'quota_id: EmbedContentRequestsPerMinute; '
                        'Please retry in 46.7s'
                    )
                return {'embedding': [[.1, .2, .3]] * len(kwargs['content'])}

            sleep = Mock()
            count = build_embeddings(root, embed, sleep_fn=sleep, progress=Mock())
            self.assertEqual(count, len(self.retriever['chunks']))
            self.assertEqual(call_count, (count + 19) // 20 + 1)
            self.assertGreaterEqual(sleep.call_count, 2)
            self.assertEqual(sleep.call_args_list[0].args[0],
                             EMBEDDING_WINDOW_DELAY_SECONDS)

    def test_embedding_builder_resumes_after_interrupted_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'FAQ_Chatbot_100.md').write_bytes(self.data)
            calls = 0

            def interrupted(**kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('interrupted')
                return {'embedding': [[.1, .2, .3]] * len(kwargs['content'])}

            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                build_embeddings(root, interrupted, sleep_fn=Mock(), progress=Mock())
            self.assertTrue((root / '.faq_embedding_checkpoint.npz').exists())

            embed = Mock(side_effect=lambda **kw: {
                'embedding': [[.1, .2, .3]] * len(kw['content'])
            })
            progress = Mock()
            count = build_embeddings(root, embed, sleep_fn=Mock(), progress=progress,
                                     pause_at_half=True)
            self.assertEqual(embed.call_count, (count - 20 + 19) // 20)
            progress.assert_any_call(f'Resuming embeddings at 20/{count} FAQs')
            vectors, warning = rag.load_persisted_embeddings(root, self.data, count)
            self.assertEqual(vectors.shape, (count, 3))
            self.assertIsNone(warning)
            self.assertFalse((root / '.faq_embedding_checkpoint.npz').exists())
            self.assertFalse((root / '.faq_embedding_checkpoint.json').exists())

    def test_rate_limit_retry_delay_ignores_unrelated_errors(self):
        self.assertIsNone(rate_limit_retry_delay(RuntimeError('network')))

    def test_rate_limit_retry_delay_supports_current_sdk(self):
        class ClientError(Exception):
            code = 429
            status = 'RESOURCE_EXHAUSTED'

        delay = rate_limit_retry_delay(
            ClientError('Quota exceeded; Please retry in 46.7s')
        )
        self.assertEqual(delay, EMBEDDING_WINDOW_DELAY_SECONDS)
        self.assertEqual(
            rate_limit_retry_delay(ClientError('Resource exhausted. Please try again later.')),
            EMBEDDING_WINDOW_DELAY_SECONDS,
        )

    def test_embedding_failure_and_cooldown(self):
        retriever = dict(self.retriever, semantic=np.ones((len(self.retriever['chunks']), 3)))
        embed = Mock(side_effect=RuntimeError('quota'))
        state = {}
        for now in (100, 101):
            result = rag.retrieve('อธิบาย', retriever, state, embed, now=now)
            self.assertEqual(result['route'], 'B')
            self.assertEqual(result['method'], 'TF-IDF fallback')
        self.assertEqual(embed.call_count, 1)
        rag.retrieve('อธิบาย', retriever, state, embed, now=161)
        self.assertEqual(embed.call_count, 2)

    def test_hybrid_and_missing_embeddings(self):
        vectors = np.ones((len(self.retriever['chunks']), 3))
        embed = Mock(return_value={'embedding': [1., 1., 1.]})
        result = rag.retrieve('อธิบาย', dict(self.retriever, semantic=vectors), {}, embed)
        self.assertEqual(result['route'], 'C')
        self.assertEqual(len(result['matches']), 3)
        embed.assert_called_once()
        embed.reset_mock()
        result = rag.retrieve('อธิบาย', dict(self.retriever, semantic=None), {}, embed)
        self.assertEqual(result['route'], 'B')
        embed.assert_not_called()
        self.assertTrue(np.isfinite(rag.normalize_scores([.5, .5])).all())

    def test_history_and_original_question(self):
        messages = [{'role': 'model', 'content': 'welcome'}]
        for index in range(10):
            messages += [{'role': 'user', 'content': str(index)}, {'role': 'model', 'content': 'answer'}]
        history = rag.build_history(messages)
        self.assertEqual(len(history), rag.MAX_HISTORY_MESSAGES)
        self.assertEqual([m['role'] for m in history], ['user', 'model'] * 3)
        question = 'ค่าเล่าเรียนเท่าไหร่'
        prompt = rag.build_rag_prompt('context', question)
        self.assertEqual(prompt.count(question), 1)
        self.assertNotIn(rag.expand_query(question), prompt)

    def test_streaming_and_legacy(self):
        class Chunk:
            def __init__(self, text): self.text = text
        class Blocked:
            @property
            def text(self): raise ValueError('no text')
        class Chat:
            def send_message(self, prompt, stream=False):
                self.calls = getattr(self, 'calls', 0) + 1
                return iter([Blocked(), Chunk('Hello'), Chunk(' world')])
        class Legacy:
            def send_message(self, prompt): return Chunk('legacy')
        class Current:
            def send_message_stream(self, prompt):
                return iter([Chunk('new '), Chunk('sdk')])
        update = Mock()
        chat = Chat()
        self.assertEqual(rag.generate_answer(chat, 'Q', update), ('Hello world', True))
        self.assertEqual(chat.calls, 1)
        update.assert_called_with('Hello world')
        self.assertEqual(rag.generate_answer(Legacy(), 'Q', update), ('legacy', False))
        self.assertEqual(rag.generate_answer(Current(), 'Q', update), ('new sdk', True))


if __name__ == '__main__':
    unittest.main()
