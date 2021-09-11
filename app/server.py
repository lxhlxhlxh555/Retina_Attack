from .utils.attacker import build_attacker
from .utils.evaluator import build_evaluator
from .utils.dataset import build_dataset
from .utils.attack_tools import load_model
from flask import Flask,request,jsonify
from flask_restful import  Api, Resource
import json
from .celery import model_test

# app = Flask(__name__)
# CORS(app,supports_credentials=True)
# api = Api(app)

class Hello(Resource):
    def get(self):
        return 'Hello World!'

class Test(Resource):
    def get(self):
        return "Robustness Test"
    def post(self):
        params = json.loads(request.data)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        task = model_test.apply_async(args=[params])
        print(task.id)
        return str('Task Id:'+task.id)

class QueryTest(Resource):
    def post(self,_id):
        task = model_test.AsyncResult(_id)
        if task.state == 'PENDING':
            #job did not start yet
            response = {
            'state': task.state,
            'current': 0,
            'success': 0,
            'total': 1,
            'status': 'Pending...'
            }
        elif task.state != 'FAILURE':
            response = {
            'state': task.state,
            'current': task.info.get('current',0),
            'success': task.info.get('success', 1),
            'total': task.info.get('total', 2),
            'fail_to_read':task.info.get('fail_to_read',[]),
            'fail_to_attack':task.info.get('fail_to_attack',[]),
            'status': task.info.get('status', '')
            }
            if 'result' in task.info:
                response['result'] = task.info['result']
        else:
            # something went wrong in the background job
            response = {
            'state': task.state,
            'current': 1,
            'success':1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
            }
        return response

# api.add_resource(Hello, '/')
# api.add_resource(NormalAttack, '/api/norm')
# api.add_resource(AdversarialAttack, '/api/adv')

# if __name__ == '__main__':
#     server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
#     server.serve_forever()
#     app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000,debug=True)


