import unittest

import altair as alt

from scenario_model.config import LABELS, NUSVR_FEATURES
from scenario_model.visualization import contribution_chart


class ContributionChartTests(unittest.TestCase):
    def test_signed_chart_keeps_all_features_and_original_values(self):
        means = {f: i / 4 for i, f in enumerate(NUSVR_FEATURES)}
        contributions = {f: (i - 4) * .851 for i, f in enumerate(NUSVR_FEATURES)}
        original_means, original_contributions = means.copy(), contributions.copy()
        chart = contribution_chart(means, contributions)
        alt.LayerChart.from_dict(chart, validate=True)
        rows = chart['data']['values']
        self.assertEqual(len(rows), 9)
        self.assertEqual([r['contribution'] for r in rows], sorted(contributions.values(), reverse=True))
        by_label = {r['feature']: r for r in rows}
        for f in NUSVR_FEATURES:
            self.assertEqual(by_label[LABELS[f]]['score'], means[f])
            self.assertEqual(by_label[LABELS[f]]['contribution'], contributions[f])
        self.assertEqual({r['direction'] for r in rows}, {'Повышает MD', 'Снижает MD', 'Нейтрально'})
        domain = chart['layer'][1]['encoding']['x']['scale']['domain']
        self.assertEqual(domain[0], -domain[1])
        self.assertGreater(domain[1], max(abs(v) for v in contributions.values()))
        self.assertEqual(means, original_means)
        self.assertEqual(contributions, original_contributions)

    def test_all_zero_or_tiny_contributions_have_neutral_labels_and_valid_scale(self):
        means = {f: 0 for f in NUSVR_FEATURES}
        for value in [0, .00001, -.00001]:
            chart = contribution_chart(means, dict.fromkeys(NUSVR_FEATURES, value))
            alt.LayerChart.from_dict(chart, validate=True)
            self.assertTrue(all(r['direction'] == 'Нейтрально' and r['display'] == '0 п.п.' for r in chart['data']['values']))
            self.assertEqual(chart['layer'][1]['encoding']['x']['scale']['domain'], [-1, 1])


if __name__ == '__main__':
    unittest.main()
