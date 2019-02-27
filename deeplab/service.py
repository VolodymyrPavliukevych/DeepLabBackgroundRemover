from deeplab import DeepLabModel
import os
from io import BytesIO
from PIL import Image
import numpy as np
import time
import re
import posixpath
import urllib.request, urllib.parse, urllib.error
import mimetypes as memetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from time import gmtime, strftime

HOST_NAME = '192.168.80.38'
PORT_NUMBER = 9000


class MyHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        paths = {
            '/upload': {'status': 200}
        }
        file_path, info_message = self.deal_post_data()
        print('File path: ', file_path)
        print('Info: ', info_message)
        if self.path in paths:
            self.respond_file(paths[self.path], file_path)
            # self.respond(paths[self.path])
        else:
            self.respond({'status': 500})

    def do_GET(self):
        paths = {
            '/upload': {'status': 200},
            '/bar': {'status': 302},
            '/baz': {'status': 404},
            '/qux': {'status': 500}
        }

        if self.path in paths:
            self.respond(paths[self.path])
        else:
            self.respond({'status': 500})

    def handle_http(self, status_code, path):
        print()
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = '''
            <!DOCTYPE html>
            <html>
            <body>

            <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file" id="file">
            <input type="submit" value="Upload Image" name="submit">
            </form>
            
            </body>
            </html>
        '''
        return bytes(content, 'UTF-8')

    def handle_file(self, status_code, path, file_path):
        self.send_response(status_code)
        content, encoding = memetypes.MimeTypes().guess_type(file_path)
        info = os.stat(file_path)
        print(content, encoding, info)
        self.send_header("Content-Type", content)
        self.send_header("Content-Encoding", encoding)
        self.send_header("Content-Length", info.st_size)
        self.end_headers()
        
        return

    def respond(self, opts):
        response = self.handle_http(opts['status'], self.path)
        self.wfile.write(response)

    def respond_file(self, opts, file_path):
        file = open(file_path, 'rb')
        original_im = Image.open(BytesIO(file.read()))
        image, seg_map = MODEL.run(original_im)
        a_channel = Image.new('L', image.size, 255)
        image.putalpha(a_channel)
        image_width, image_height = image.size
        for x in range(image_width):
          for y in range(image_height):
            rgba = image.getpixel((x, y))
            value = seg_map[y][x]
            alpha = 0 if value == 0 else 255
            rgba = (rgba[0], rgba[1], rgba[2], int(alpha))
            image.putpixel((x, y), rgba)
        result_image_path = 'upload/' + strftime("%H%M%s", gmtime()) + '.png' 
        image.save(result_image_path)

        self.handle_file(opts['status'], self.path, result_image_path)
        result_file = open(result_image_path, 'rb')
        self.wfile.write(result_file.read())

    def deal_post_data(self):
        content_type = self.headers['content-type']
        if not content_type:
            return (None, "Content-Type header doesn't contain boundary")
        boundary = content_type.split("=")[1].encode()
        remainbytes = int(self.headers['content-length'])
        line = self.rfile.readline()
        remainbytes -= len(line)
        if not boundary in line:
            return (None, "Content NOT begin with boundary")
        line = self.rfile.readline()
        remainbytes -= len(line)
        fn = re.findall(r'Content-Disposition.*name="file"; filename="(.*)"', line.decode())
        if not fn:
            return (None, "Can't find out file name...")
        path = self.translate_path(self.path)
        fn = os.path.join(path, fn[0])
        line = self.rfile.readline()
        remainbytes -= len(line)
        line = self.rfile.readline()
        remainbytes -= len(line)
        print('Write at: ', fn)
        try:
            out = open(fn, 'wb')
        except IOError:
            return (None, "Can't create file to write, do you have permission to write?")
                
        preline = self.rfile.readline()
        remainbytes -= len(preline)
        while remainbytes > 0:
            line = self.rfile.readline()
            remainbytes -= len(line)
            if boundary in line:
                preline = preline[0:-1]
                if preline.endswith(b'\r'):
                    preline = preline[0:-1]
                out.write(preline)
                out.close()
                return (fn, "File '%s' upload success!" % fn)
            else:
                out.write(preline)
                preline = line
        return (None, "Unexpect Ends of data.")

    def translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax.
        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored.  (XXX They should
        probably be diagnosed.)
        """
        # abandon query parameters
        path = path.split('?',1)[0]
        path = path.split('#',1)[0]
        path = posixpath.normpath(urllib.parse.unquote(path))
        words = path.split('/')
        words = [_f for _f in words if _f]
        path = os.getcwd()
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir): continue
            path = os.path.join(path, word)
        return path        

MODEL = DeepLabModel('/server/repository/models/research/deeplab/model/deeplabv3_mnv2_pascal_train_aug/')

if __name__ == '__main__':
    server_class = HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))

