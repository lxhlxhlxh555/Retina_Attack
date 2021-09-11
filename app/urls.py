from flask_restful import Resource, Api
from .server import Hello, Test, QueryTest #导入路由的具体实现方法
from flask import Blueprint

#创建一个蓝图对象
assets_page = Blueprint('assets_page', __name__)
#在这个蓝图对象上进行操作,注册路由
api = Api(assets_page)

#注册路由
api.add_resource(Hello,'/')
# api.add_resource(NormalAttack,'/api/norm/')
# api.add_resource(QueryNorm,'/api/norm/<_id>')
# api.add_resource(AdversarialAttack,'/api/adv/')
# api.add_resource(QueryAdv,'/api/adv/<_id>')
api.add_resource(Test,'/api/test/')
api.add_resource(QueryTest,'/api/test/<_id>')