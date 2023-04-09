# 导入 Flask 类
import random
import string
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED ,TimeoutError
from flask import Flask, request
from flask_cors import *
import json, os, shutil, sys
from threading import Thread
import threading, subprocess, signal
import recgnize.recgnizeFace as rF
from multiprocessing import Process
from numba import cuda
# 创建了这个类的实例。第一个参数是应用模块或者包的名称。
basedir = os.path.abspath(os.path.dirname(__file__))
current_encoding = 'utf-8'

Model = rF.FaceNet()
LockImg = threading.Lock()
LockVideo = threading.Lock()

app = Flask(__name__, static_folder='alphapose_results/',static_url_path="/alphapose_results")
CORS(app, supports_credentials=True)
# 使用 route() 装饰器来告诉 Flask 触发函数的 URL

class MyThread(Thread):
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args= args
        self.result = None    

    def run(self):
        self.result = self.func(*self.args)
    
    def getResult(self):
        return self.result

def run_cmd(cmd_string, timeout):
    os.system("export PATH=/usr/local/cuda/bin/:$PATH")
    os.system("export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH")
    p = subprocess.Popen("cd ./AlphaPose\n "+cmd_string + "\n cd ..", stderr=subprocess.STDOUT,stdout=subprocess.PIPE, shell=True, close_fds=True,
                         preexec_fn=os.setsid)
    while p.poll() is None:
        line=p.stdout.readline().decode("utf8")
        print(line)
    try:
        msg, errs = p.communicate(timeout=timeout)
        ret_code = p.poll()
        if ret_code:
            code = 1
            msg = "[Error]Called Error ： " + str(msg.decode('utf-8'))
        else:
            code = 0
            msg = str(msg.decode('utf-8'))
    except subprocess.TimeoutExpired:
        #不能只使用p.kill和p.terminate，无法杀干净所有的子进程，需要使用os.killpg
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGTERM)
       
        code = 1
        msg = "[ERROR]Timeout Error : Command '" + cmd_string + "' timed out after " + str(timeout) + " seconds"
    return code, msg


def ThreadImg(runcmd, outpath):
    LockImg.acquire() #加线程锁
    if (os.path.exists (outpath)):
        shutil.rmtree(outpath)
    code, msg = run_cmd(runcmd, 30)
    LockImg.release() #解锁
    return code

def ThreadVideo(runcmd, outpath, imgName, uname):
    LockVideo.acquire()
    if (os.path.exists (outpath)):
        shutil.rmtree(outpath)
    code, msg = run_cmd(runcmd, 130)
    if (code != 0):
        return code
    code = os.system("ffmpeg -i "+'alphapose_results/'+uname+'/video/'+'AlphaPose_'+imgName +" " + 'alphapose_results/'+uname+'/video/'+'Result_'+imgName)
    LockVideo.release()
    return code

@app.route("/uploadImg", methods=["POST"])
@cross_origin()
def AlphaPoseImg():
    if request.method == "POST":
        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))
    #获取图片文件 name = file
    img = request.files.get('file')
    path = basedir+'/source/' + request.form['uname'] + '/img/'
    imgName = ran_str+os.path.splitext(img.filename)[-1]
    file_path = path+imgName
    if (os.path.exists (path)):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    img.save(file_path)
    cmd = []
    cmd.append("python "+ "scripts/demo_inference.py")
    cfg = "--cfg "+basedir+"/AlphaPose/configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml"
    checkpoint = "--checkpoint "+ basedir+"/AlphaPose/pretrained_models/halpe136_fast50_regression_256x192.pth"
    indir = "--indir "+ path
    outdir = "--outdir " + basedir + "/alphapose_results/" + request.form['uname'] + '/img/'
    outpath = basedir + "/alphapose_results/" + request.form['uname'] + '/img/'
    cmd.append(cfg)
    cmd.append(checkpoint)
    cmd.append(indir)
    cmd.append(outdir)
    cmd.append("--save_img")
    runcmd = ' '.join(cmd)
    #print(runcmd)
    
    func1 = MyThread(ThreadImg, args =(runcmd, outpath))
    func1.start()
    func1.join(30)


    code = func1.getResult()

    if (code != 0): 
        msg = "Failed!"
    else:
        msg = "Success!"
    dict_result = {
        "code":code,
        "data":'alphapose_results/'+request.form['uname']+'/img/vis/'+imgName,
        "msg":msg
    }
    return dict_result

@app.route("/uploadVideo", methods=["POST"])
@cross_origin()
def AlphaPoseVideo():
    if request.method == "POST":
        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))
    #获取图片文件 name = file
    img = request.files.get('file')
    path = basedir+'/source/' + request.form['uname'] + '/video/'
    imgName = ran_str+os.path.splitext(img.filename)[-1]
    file_path = path+imgName
    if (os.path.exists (path)):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    img.save(file_path)
    
    cmd = []
    cmd.append("python "+ "scripts/demo_inference.py")
    cfg = "--cfg "+basedir+"/AlphaPose/configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml"
    checkpoint = "--checkpoint "+ basedir+"/AlphaPose/pretrained_models/multi_domain_fast50_regression_256x192.pth"
    video = "--video "+ file_path
    outdir = "--outdir " + basedir + "/alphapose_results/" + request.form['uname'] + '/video/'
    outpath = basedir + "/alphapose_results/" + request.form['uname'] + '/video/'
    cmd.append(cfg)
    cmd.append(checkpoint)
    cmd.append(video)
    cmd.append(outdir)
    cmd.append("--save_video")
    if (os.path.getsize(file_path) > 1024*1024):
        cmd.append("--vis_fast")
    runcmd = ' '.join(cmd)
    print(runcmd)
    func1 = MyThread(ThreadVideo, args =(runcmd, outpath, imgName, request.form['uname']))
    func1.start()
    func1.join(140)
    code = func1.getResult()
    if (code != 0): 
        msg = "Failed!"
    else:
        msg = "Success!"
    dict_result = {
        "code":code,
        "data":'alphapose_results/'+request.form['uname']+'/video/'+'Result_'+imgName,
        "msg":msg
    }
    return dict_result



@app.route("/saveFaceImg", methods=["POST"])
@cross_origin()
def saveFaceImg():
    global Model
    upload_files=request.files.getlist('files')
    outpath = basedir + "/source/" + "faceImg/"+ request.form['uname']+'/'
    if (os.path.exists (outpath)):
        shutil.rmtree(outpath)
    os.makedirs(outpath, exist_ok=True)
    for file in upload_files:
        upload_path = os.path.join(outpath,file.filename)
        file.save(upload_path)
    newProgress = Process(target=Model.buildModel())
    #Model.buildModel()
    newProgress.start()
    newProgress.join()
    return "success"


def recTime(image):
    global Model
    return Model.compare(image)

@app.route("/recgnizeface", methods=["POST"])
@cross_origin()
def recgnizeface():
    img = request.files.get('image')
    image = img.read()
    #获取图片文件 name = file
    obj_list=[]
    with ThreadPoolExecutor(max_workers=10) as t:
        result = t.submit(recTime, image)
        obj_list.append(result)
        wait(obj_list, return_when = ALL_COMPLETED)
    print(result.result())
    return result.result()





if __name__ == "__main__":
    app.run(debug=True,port=int("6006"), use_reloader=False)
