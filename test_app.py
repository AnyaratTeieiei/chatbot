"""Streamlit integration checks, with Gemini mocked and no network calls."""

from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from rag import CONTEXT_STATE_KEY, NOT_FOUND


ROOT = Path(__file__).resolve().parent


class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.env = patch.dict("os.environ", {"GEMINI_API_KEY_INSURVERSE": "offline-test-placeholder"})
        self.env.start()
        self.dotenv = patch("dotenv.load_dotenv")
        self.dotenv.start()
        self.client_factory = patch("gemini_service.create_client")
        self.client = self.client_factory.start().return_value
        self.client.models.embed_content.side_effect = RuntimeError("offline")
        self.chat = self.client.chats.create.return_value
        self.chat.send_message_stream.return_value = iter([SimpleNamespace(text="คำตอบทดสอบ")])
        self.addCleanup(patch.stopall)

    def app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        self.assertFalse(app.exception)
        return app

    def test_startup_and_curriculum_answer(self):
        app = self.app()
        self.client.chats.create.assert_not_called()
        app.chat_input[0].set_value("หลักสูตรนี้เรียนกี่ปี").run()
        self.assertFalse(app.exception)
        self.chat.send_message_stream.assert_called_once()
        question = self.chat.send_message_stream.call_args.args[0]
        self.assertIn("## หน้า PDF 005", question)
        self.assertIn("หลักสูตรนี้เรียนกี่ปี", question)
        self.assertNotIn("## FAQ", question)
        self.assertEqual(app.session_state["messages"][-1]["content"], "คำตอบทดสอบ")

    def test_blank_and_current_only_questions(self):
        app = self.app()
        original = len(app.session_state["messages"])
        app.chat_input[0].set_value("   ").run()
        self.assertEqual(len(app.session_state["messages"]), original)
        app.chat_input[0].set_value("ค่าเทอมปี 2569 เท่าไหร่").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["messages"][-1]["content"], NOT_FOUND)
        self.chat.send_message_stream.assert_not_called()
        self.assertNotIn(CONTEXT_STATE_KEY, app.session_state)

    def test_follow_up_and_clear(self):
        app = self.app()
        app.chat_input[0].set_value("หลักสูตรนี้มีกี่แขนงวิชาให้เลือก").run()
        self.assertFalse(app.exception)
        self.assertIn(CONTEXT_STATE_KEY, app.session_state)
        self.chat.send_message_stream.reset_mock()
        self.chat.send_message_stream.return_value = iter([SimpleNamespace(text="คำตอบต่อเนื่อง")])
        app.chat_input[0].set_value("อะไรบ้าง").run()
        self.assertFalse(app.exception)
        prompt = self.chat.send_message_stream.call_args.args[0]
        self.assertIn("แขนงวิชา", prompt)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertNotIn(CONTEXT_STATE_KEY, app.session_state)

    def test_generation_failure_shows_sourced_excerpt(self):
        def broken():
            yield SimpleNamespace(text="partial")
            raise RuntimeError("simulated")

        self.chat.send_message_stream.return_value = broken()
        app = self.app()
        app.chat_input[0].set_value("หลักสูตรนี้เรียนกี่ปี").run()
        self.assertFalse(app.exception)
        answer = app.session_state["messages"][-1]["content"]
        self.assertNotIn("partial", answer)
        self.assertIn("หน้า PDF 5", answer)
        self.assertIn("เล่มหลักสูตร", answer)

    def test_without_api_key_uses_sourced_excerpt(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY_INSURVERSE": ""}), patch("streamlit.secrets", {}):
            app = self.app()
            app.chat_input[0].set_value("หลักสูตรนี้เรียนกี่ปี").run()
            self.assertFalse(app.exception)
            self.assertIn("หน้า PDF 5", app.session_state["messages"][-1]["content"])
            self.chat.send_message_stream.assert_not_called()

    def test_broad_course_questions_answer_without_generation_and_do_not_repeat(self):
        app = self.app()
        app.chat_input[0].set_value("เรียนวิชาอะไรบ้าง").run()
        overview = app.session_state["messages"][-1]["content"]
        self.assertIn("175 หน่วยกิต", overview)
        app.chat_input[0].set_value("วิชาศึกษาทั่วไปมีอะไรบ้าง").run()
        general = app.session_state["messages"][-1]["content"]
        self.assertIn("5 กลุ่ม", general)
        self.assertNotEqual(overview, general)
        self.chat.send_message_stream.assert_not_called()

    def test_temporary_generation_error_uses_sourced_fallback(self):
        class TemporaryError(RuntimeError):
            code = 503

        self.chat.send_message_stream.side_effect = TemporaryError("temporarily unavailable")
        app = self.app()
        app.chat_input[0].set_value("หลักสูตรนี้เรียนกี่ปี").run()
        self.assertFalse(app.exception)
        self.chat.send_message_stream.assert_called_once()
        self.assertIn("หน้า PDF 5", app.session_state["messages"][-1]["content"])

if __name__ == "__main__":
    unittest.main()
