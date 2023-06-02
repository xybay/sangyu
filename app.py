import os
import zipfile
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from main_process import main_process
from datetime import datetime
from datetime import timedelta
import logging
import time

upload_time = ''

app = Flask(__name__, static_folder='dist')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'zip'}

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)



# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, '_index.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    global upload_time
    upload_time = str(datetime.now().strftime('%Y%m%d%H%M%S'))
    filename = 'data_' + upload_time + '.zip'
    if not os.path.exists('user_data_zip'):
        os.makedirs('user_data_zip')
    file.save(os.path.join('user_data_zip', filename))
    extract_path = os.path.join('user_data', 'dcm_' + upload_time)
    with zipfile.ZipFile(os.path.join('user_data_zip', filename), 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return '数据上传完成，请点击“开始处理”按钮'

@app.route('/process_start', methods=['POST'])
def process_start():
    extract_path = os.path.join('/home/kangxy/user_data', 'dcm_' + upload_time)
    result, acc = main_process(extract_path)
    # time.sleep(5)
    # result = 'NC'
    # acc = '97.7%'
    return jsonify({'result': result, 'acc': acc})



if __name__ == '__main__':
    # from werkzeug.middleware.proxy_fix import ProxyFix
    # app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='0.0.0.0', port=8001, debug=True)# 202.204.62.78
    # app.run()


# sudo lsof -i :8001
# sudo systemctl reload nginx
# nginx -t 
