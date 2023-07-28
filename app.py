from flask import Flask, request, jsonify
from semantic_similarity_calculator import SemanticSimilarityCalculator

app = Flask(__name__)
similarity_calculator = SemanticSimilarityCalculator()

@app.route('/calculate_similarity', methods=['GET'])
def calculate_similarity():
    try:
        text1 = request.args.get('text1')
        text2 = request.args.get('text2')

        if not text1 or not text2:
            return jsonify({'error': 'Both text1 and text2 parameters are required.'}), 400

        similarity_score = similarity_calculator.calculate_semantic_similarity(text1, text2)

        response = {'Similarity Score': similarity_score.item()}  # Convert to Python float
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()
