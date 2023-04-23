import unittest
from cosine_similarity import CosineSimilarity


class TestCosineSimilarity(unittest.TestCase):

    def setUp(self):
        self.cosine_similarity = CosineSimilarity()

    def test_similarity_between_two_lists(self):
        sentences1 = ['The cat sits outside', 'A man is playing guitar', 'The new movie is awesome']
        sentences2 = ['The dog plays in the garden', 'A woman watches TV', 'The new movie is so great']

        results = self.cosine_similarity.similarity_between_two_lists(sentences1, sentences2)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            self.assertIsInstance(result[0], str)
            self.assertIsInstance(result[1], str)
            self.assertIsInstance(result[2], float)

    def test_top_similar_pairs(self):
        sentences = ['The cat sits outside', 'A man is playing guitar', 'I love pasta', 'The new movie is awesome',
                     'The cat plays in the garden', 'A woman watches TV', 'The new movie is so great', 'Do you like pizza?']

        results = self.cosine_similarity.top_similar_pairs(sentences)
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            self.assertIsInstance(result[0], str)
            self.assertIsInstance(result[1], str)
            self.assertIsInstance(result[2], float)

    def test_semantic_search(self):
        corpus = ['A man is eating food.', 'A man is eating a piece of bread.', 'The girl is carrying a baby.',
                  'A man is riding a horse.', 'A woman is playing violin.', 'Two men pushed carts through the woods.',
                  'A man is riding a white horse on an enclosed ground.', 'A monkey is playing drums.', 'A cheetah is running behind its prey.']
        queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']

        for query in queries:
            results = self.cosine_similarity.semantic_search(corpus, query)
            self.assertEqual(len(results), 5)
            for result in results:
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                self.assertIsInstance(result[0], str)
                self.assertIsInstance(result[1], float)


if __name__ == '__main__':
    unittest.main()
