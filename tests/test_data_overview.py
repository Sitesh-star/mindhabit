import io
import json
import re
import shutil
import subprocess
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

import app as mindhabits
import test_spreadsheets as spreadsheet_tests
from utils.dashboard_analyzer import build_dashboard_content, prepare_dashboard_dataframe


class DataOverviewTests(unittest.TestCase):
    # Reuse the existing isolated database/upload fixture without inheriting its tests.
    setUp = spreadsheet_tests.SpreadsheetTests.setUp
    tearDown = spreadsheet_tests.SpreadsheetTests.tearDown
    save = spreadsheet_tests.SpreadsheetTests.save

    def chart_data(self, page):
        return json.loads(re.search(r'<script id="distributionData" type="application/json">(.*?)</script>', page, re.S)[1])

    def test_csv_overview_raw_records_and_next_step(self):
        rows = "User ID,Date,Sleep Hours,Stress Score,Mood,Steps,Screen Hours,Device Model\n"
        rows += "".join(f"{i},2026-01-{i + 1:02},7,{i % 4},Good,{4000 + i * 100},3,Phone-X\n" for i in range(15))
        with patch("app._direct_blob_upload_enabled", return_value=False):
            response = self.client.post('/upload', data={'file': (io.BytesIO(rows.encode()), 'wellness.csv')})
        self.assertEqual(response.location, '/dashboard')
        page = self.client.get(response.location).get_data(as_text=True)
        for text in ['Data Overview', 'wellness.csv', '15', '100%', '01 Jan 2026', '15 Jan 2026',
                     'Your Data at a Glance', 'Key Patterns', 'Average 7 hours', 'View Your Records',
                     'showing the first 10 of 15 records', 'Analyse My Well-Being']:
            self.assertIn(text, page)
        self.assertRegex(page, r'href="/wellbeing"[^>]*>Analyse My Well-Being')
        self.assertIn('<details class="overview-records">', page)
        self.assertNotRegex(page, r'<details[^>]*\bopen\b')
        table = re.search(r'<tbody>(.*?)</tbody>', page, re.S)[1]
        self.assertEqual(table.count('<tr>'), 10)
        self.assertIn('<th scope="col">User ID</th>', page)
        self.assertEqual(self.client.get('/preview').data, self.client.get('/dashboard').data)
        charts = self.chart_data(page)
        self.assertLessEqual(len(charts), 3)
        self.assertTrue(charts)
        self.assertTrue(all(chart['name'] not in {'user_id', 'device_model'} for chart in charts))

    def test_manual_overview_and_legacy_selection_query(self):
        self.payload = {'name': 'Manual Wellness', 'columns': ['Sleep Hours', 'Mood'],
                        'rows': [['6', 'Good'], ['8', 'Low']]}
        saved = self.save()
        self.assertEqual(saved['preview_url'], '/dashboard')
        self.assertEqual(saved['dashboard_url'], '/dashboard')
        other = self.save_other_sheet()
        response = self.client.get('/dashboard', query_string={'spreadsheet_id': saved['id']})
        page = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        for text in ['Manual Wellness', 'Manual entry', 'Average 7 hours', 'Most common:', '>6</td>', '>Good</td>']:
            self.assertIn(text, page)
        self.assertNotIn(other['name'], page)
        self.assertNotIn('Date Range', page)
        self.assertEqual(self.client.get('/dashboard?spreadsheet_id=999999').status_code, 404)

    def save_other_sheet(self):
        return self.client.post('/api/spreadsheets', json={**self.payload, 'name': 'Other sheet'}).get_json()

    def test_empty_state_and_navigation(self):
        for route in ['/dashboard', '/preview']:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200)
            page = response.get_data(as_text=True)
            for text in ['Data Overview', 'No data available yet',
                         'Upload a CSV file or enter your data manually to continue.',
                         'Upload CSV', 'Enter Data Manually']:
                self.assertIn(text, page)
        for route in ['/', '/upload', '/manual', '/dashboard', '/wellbeing', '/habits', '/export', '/help']:
            page = self.client.get(route).get_data(as_text=True)
            nav = re.search(r'<nav class="nav-menu">(.*?)</nav>', page, re.S)[1]
            self.assertEqual(nav.count('Data Overview'), 1, route)
            self.assertNotIn('Data Preview', nav)
            self.assertNotIn('<span>Dashboard</span>', nav)

    def test_unrelated_data_keeps_records_without_summaries(self):
        self.save()
        page = self.client.get('/dashboard').get_data(as_text=True)
        self.assertIn('did not detect enough wellness-related information', page)
        self.assertIn('View Your Records', page)
        self.assertIn('Custom habit', page)
        self.assertNotIn('Your Data at a Glance', page)
        self.assertEqual(self.chart_data(page), [])

    def test_manual_completeness_uses_blank_cells(self):
        self.payload = {'name': 'Partly complete', 'columns': ['Sleep Hours', 'Mood'],
                        'rows': [['6', 'Good'], ['8', '   ']]}
        self.save()
        page = self.client.get('/dashboard').get_data(as_text=True)
        self.assertIn('75%', page)
        self.assertIn('Average 7 hours', page)

    def test_overview_chart_javascript_and_user_text_are_safe(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node.js unavailable')
        dangerous = "</script><script>alert('test')</script>"
        self.payload = {'name': dangerous, 'columns': ['Mood'], 'rows': [[dangerous], ['Good']]}
        self.save()
        page = self.client.get('/dashboard').get_data(as_text=True)
        self.assertNotIn(dangerous, page)
        charts = self.chart_data(page)
        self.assertIn(dangerous, charts[0]['labels'])
        script = re.search(r'<script>(.*?)</script>', page, re.S)[1]
        path = Path(self.directory.name) / 'overview.js'
        path.write_text(script, encoding='utf-8')
        result = subprocess.run([node, '--check', str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_removes_project_section_and_returns_home(self):
        page = self.client.get('/help').get_data(as_text=True)
        self.assertNotIn('Final Year Project', page)
        self.assertNotIn('project-about', page)
        self.assertNotIn('/home.html', page)
        self.assertRegex(page, r'href="/" class="primary-button">\s*Return Home')
        self.assertIn('Important Wellness Disclaimer', page)
        self.assertIn('How to Use MindHabits', page)


class OverviewSelectionTests(unittest.TestCase):
    def content(self, columns):
        return build_dashboard_content(prepare_dashboard_dataframe(pd.DataFrame(columns)))

    def test_excludes_identifiers_encoded_categories_and_unrelated_fields(self):
        result = self.content({'User ID': [1, 2, 3], 'Sleep ID': [10, 11, 12],
                               'Mood Code': [0, 1, 2], 'Device Model': ['A', 'B', 'A'],
                               'Country': ['A', 'B', 'A'], 'Disk Usage': [1, 2, 3], 'Sleep Hours': [6, 8, 7]})
        self.assertEqual([item['name'] for item in result['column_summaries']], ['sleep_hours'])
        self.assertEqual(result['column_summaries'][0]['main'], 'Average 7 hours')

    def test_chart_titles_and_data_based_explanations(self):
        result = self.content({'sleep_duration_hrs': [6, 6, 8, 8], 'stress_score': [1, 2, 3, 4],
                               'mood': ['Neutral', 'Neutral', 'Neutral', 'Good']})
        charts = {chart['title']: chart for chart in result['distribution_charts']}
        self.assertEqual(set(charts), {'Sleep Duration', 'Stress Levels', 'Mood / Well-Being'})
        self.assertIn('between 6 and 8 hours', charts['Sleep Duration']['description'])
        self.assertIn('Neutral is the most common response (3 of 4', charts['Mood / Well-Being']['description'])
        self.assertTrue(all(chart['type'] == 'bar' for chart in charts.values()))

    def test_dates_are_optional_numeric_dates_are_not_fabricated(self):
        for dates in [[1, 2, 3, 4], ['bad'] * 4, ['2026-01-01', 'bad', 'bad', 'bad']]:
            result = self.content({'Date': dates, 'Sleep Hours': [6, 7, 8, 9]})
            self.assertIsNone(result['date_range'])
            self.assertEqual(result['distribution_charts'][0]['type'], 'bar')

    def test_short_dates_fall_back_to_distribution(self):
        result = self.content({'Date': ['2026-01-01', '2026-01-02'], 'Sleep Hours': [6, 8]})
        self.assertIsNotNone(result['date_range'])
        self.assertEqual(result['distribution_charts'][0]['type'], 'bar')

    def test_constant_values_and_single_record_remain_useful(self):
        result = self.content({'Sleep Hours': [7], 'Mood': ['Good']})
        self.assertEqual(result['wellness_column_count'], 2)
        self.assertEqual(len(result['column_summaries']), 2)
        self.assertEqual(result['distribution_charts'], [])

    def test_all_areas_counted_cards_and_charts_limited(self):
        result = self.content({name: [1, 2, 3, 4] for name in
                               ['Sleep Hours', 'Stress', 'Mood', 'Steps', 'Screen Hours',
                                'Meditation Minutes', 'Caffeine Mg', 'Alcohol', 'Recovery', 'HRV']})
        self.assertEqual(result['wellness_column_count'], 10)
        self.assertEqual(len(result['column_summaries']), 6)
        self.assertEqual(len(result['distribution_charts']), 3)

    def test_completeness_and_infinite_values_do_not_generate_invalid_json(self):
        result = self.content({'Sleep Hours': [6, float('inf'), 8, None]})
        json.dumps(result, allow_nan=False)
        self.assertEqual(result['column_summaries'][0]['main'], 'Average 7 hours')
