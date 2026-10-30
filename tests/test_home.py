import unittest

import app as mindhabits


class HomePageTests(unittest.TestCase):
    def setUp(self):
        self.previous_testing = mindhabits.app.config["TESTING"]
        mindhabits.app.config["TESTING"] = True
        self.client = mindhabits.app.test_client()

    def tearDown(self):
        mindhabits.app.config["TESTING"] = self.previous_testing

    def test_home_page_contains_only_the_simplified_user_content(self):
        response = self.client.get("/")
        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for text in [
            "Understand Your Habits.",
            "Improve Your Well-Being.",
            "Explore patterns in your lifestyle and wellness data.",
            "Start Analysis",
            "Enter Data Manually",
            "Add Your Data",
            "Analyse Your Patterns",
            "View Your Insights",
            "Wellness Disclaimer",
        ]:
            self.assertIn(text, page)

        self.assertIn('href="/upload"', page)
        self.assertIn('href="/manual"', page)
        for removed_text in [
            "Mood Categories",
            "Behavioral Domains",
            "Data-Based Analysis",
            "What Can You Explore?",
            "Physiological Data",
        ]:
            self.assertNotIn(removed_text, page)

        for technical_text in [
            "Logistic Regression", "Random Forest", "XGBoost",
            "target variable", "feature importance",
        ]:
            self.assertNotIn(technical_text, page)


if __name__ == "__main__":
    unittest.main()
