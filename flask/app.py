from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/conventional')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/image-result', methods=['GET', 'POST'])
def image_result():
    try:
        original_title = request.form['original-title']
        nbrs = model.show_recommendations(original_title, image_df)
        query = nbrs[0]
        recs = nbrs[1:]    
        return render_template('image-result.html', query=query, recs=recs)
    except:
        return render_template('error.html')


if __name__ == '__main__':
    image_df = model.load_pickled_df()
    app.run(debug=True)
