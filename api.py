import os
from flask import Flask, request, redirect, url_for, jsonify, send_file
app = Flask(__name__)
from parse_data import predict

ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = './files'

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
@app.route("/")
def hello():
    return "Hello World!"


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print('ok')
        if 'file' not in request.files:
            print('ok3')
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print (2)
        if file.filename == '':
            print('ok2')
            flash('No selected file')
            return redirect(request.url)
        print(file.name)
        if file:
            print('ok1')
            print(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            NAME,FOLLOWER, LIKE, TOTAL_LIKE, TOTAL_RETWEET, count_it, kq = predict('./files/'+file.filename)

            #result = parse('./files/'+ file.filename)
            t = ''
            if kq[0] == 'it':
                t = 'Co anh huong'
            else:
                t = 'Khong co anh huong'
            return jsonify({'LINK':NAME,'FOLLOWER':FOLLOWER, 'LIKE':LIKE, 'TOTAL_LIKE':TOTAL_LIKE, 'TOTAL_RETWEET':TOTAL_RETWEET, 'count_it':count_it, 'kq': t})
        print(4)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

app.run()
