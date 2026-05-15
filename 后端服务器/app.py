"""
乡镇服务反馈平台 - 后端服务器主程序
使用 Whisper 实现语音识别 + Qwen 语义修正
"""

import os
import uuid
import requests
import json
import base64
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import secrets
import re
from functools import wraps
import zhconv
import threading

# ==================== 设置缓存目录到 D 盘 ====================
os.environ['HF_HOME'] = 'D:/township_project/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/township_project/huggingface_cache'
os.environ['XDG_CACHE_HOME'] = 'D:/township_project/cache'
os.environ['WHISPER_CACHE_DIR'] = 'D:/township_project/whisper_cache'

# ==================== 添加SSL警告抑制 ====================
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== Whisper ASR 导入 ====================
try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
    print("✅ Whisper ASR 导入成功")
    print(f"   PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
except ImportError as e:
    WHISPER_AVAILABLE = False
    print(f"❌ Whisper 导入失败: {e}")
    print("请运行: pip install openai-whisper")

# ==================== 魔搭社区配置 ====================
MODELSCOPE_API_KEY = "ms-16aba1de-ac32-4c67-b4e1-a87d588dc217"  
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"

# ==================== 初始化扩展 ====================
db = SQLAlchemy()
jwt = JWTManager()

# 配置文件上传
UPLOAD_FOLDER = 'D:/township_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'm4a'}

def allowed_file(filename, file_types):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in file_types

# ==================== 粤语微调模型配置 ====================
FINETUNED_MODEL_PATH = r"D:\township_project\后端服务器\whisper_training\models\whisper-canto-small\final"
FALLBACK_MODEL_PATH = r"D:\township_project\后端服务器\whisper_training\models\whisper-canto-small\final"

# 全局变量存储微调模型
_finetuned_model = None
_finetuned_processor = None

def load_finetuned_whisper_model():
    """加载微调后的粤语模型"""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import os
    
    # 优先使用完整训练的 medium 模型
    if os.path.exists(FINETUNED_MODEL_PATH):
        try:
            print(f"🎯 加载粤语微调模型: {FINETUNED_MODEL_PATH}")
            processor = WhisperProcessor.from_pretrained(FINETUNED_MODEL_PATH)
            model = WhisperForConditionalGeneration.from_pretrained(FINETUNED_MODEL_PATH)
            model.eval()
            return model, processor
        except Exception as e:
            print(f"⚠️ 加载微调模型失败: {e}")
    
    # 降级使用快速测试模型
    if os.path.exists(FALLBACK_MODEL_PATH):
        try:
            print(f"🎯 加载粤语微调模型(测试版): {FALLBACK_MODEL_PATH}")
            processor = WhisperProcessor.from_pretrained(FALLBACK_MODEL_PATH)
            model = WhisperForConditionalGeneration.from_pretrained(FALLBACK_MODEL_PATH)
            model.eval()
            return model, processor
        except Exception as e:
            print(f"⚠️ 加载测试模型失败: {e}")
    
    print("⚠️ 未找到微调模型，使用原始 Whisper")
    return None, None

# 尝试加载微调模型
try:
    _finetuned_model, _finetuned_processor = load_finetuned_whisper_model()
    if _finetuned_model is not None:
        print("✅ 粤语微调模型加载成功！")
except Exception as e:
    print(f"⚠️ 粤语微调模型加载失败: {e}")
    _finetuned_model = None
    _finetuned_processor = None

# ==================== Whisper 模型缓存 ====================
_whisper_model = None
_whisper_model_loading = False
_whisper_model_load_time = None
_whisper_model_size = None

_whisper_model_lock = threading.Lock()

def get_whisper_model(model_size="base", use_finetuned=True):
    """获取 Whisper 语音识别模型（优先使用微调模型）"""
    global _whisper_model, _whisper_model_loading, _whisper_model_size, _finetuned_model
    
    # 优先使用微调模型（如果请求的是粤语识别）
    if use_finetuned and _finetuned_model is not None:
        print("🎯 使用粤语微调模型进行识别")
        return _finetuned_model
    
    if not WHISPER_AVAILABLE:
        return None
    
    with _whisper_model_lock:
        if _whisper_model is not None:
            if _whisper_model_size != model_size:
                print(f"需要切换模型: {_whisper_model_size} -> {model_size}")
                _whisper_model = None
            else:
                print(f"使用已加载的 Whisper 模型 (大小: {_whisper_model_size})")
                return _whisper_model
        
        if _whisper_model_loading:
            print("Whisper 模型正在加载中，请稍候...")
            return None
        
        _whisper_model_loading = True
    
    try:
        print(f"正在加载 Whisper {model_size} 模型...")
        print(f"模型大小: ", end="")
        if model_size == "tiny":
            print("~75MB (最快)")
        elif model_size == "base":
            print("~142MB (推荐)")
        elif model_size == "small":
            print("~466MB (较准确)")
        elif model_size == "medium":
            print("~1.5GB (很准确)")
        elif model_size == "large":
            print("~2.9GB (最准确)")
        
        start_time = time.time()
        
        _whisper_model = whisper.load_model(model_size)
        
        load_time = time.time() - start_time
        _whisper_model_load_time = load_time
        _whisper_model_size = model_size
        
        print(f"✅ Whisper {model_size} 模型加载成功！耗时: {load_time:.1f}秒")
        
        device = next(_whisper_model.parameters()).device
        print(f"   运行设备: {device}")
        
    except Exception as e:
        print(f"❌ Whisper 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        _whisper_model = None
        _whisper_model_size = None
    finally:
        _whisper_model_loading = False
    
    return _whisper_model

# ==================== AI服务类 ====================
class AIService:
    """AI服务（魔搭社区API调用版）"""
    
    def __init__(self):
        self.api_key = MODELSCOPE_API_KEY
        self.base_url = MODELSCOPE_BASE_URL
        self.model = "Qwen/Qwen3-14B"
        self.categories = {
            '农业服务': 'agriculture',
            '基建维修': 'infrastructure',
            '环境卫生': 'environment',
            '医疗卫生': 'health',
            '民政咨询': 'civil',
            '其他': 'general'
        }
        self.dept_names = {
            'agriculture': '农业服务部',
            'infrastructure': '基建维修部',
            'environment': '环境整治部',
            'health': '医疗卫生部',
            'civil': '民政服务部',
            'general': '综合服务部'
        }
    
    def classify_feedback(self, content):
        """调用魔搭API对反馈内容进行分类"""
        if not content or len(content.strip()) < 3:
            return self._fallback_classify(content)
            
        prompt = f"""你是一个专业的乡村事务分类助手。请将以下村民反馈分类到最合适的类别中，只返回类别名称。
类别列表：{', '.join(self.categories.keys())}

反馈内容：{content}

分类结果："""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个精准的乡村事务分类助手，专门处理村民关于农业、基建、环境、医疗、民政等方面的反馈。只返回类别名称，不要任何解释。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 20,
            "enable_thinking": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=10,
                verify=False
            )
            
            if response.status_code != 200:
                print(f"API调用失败: {response.status_code}")
                print(f"响应内容: {response.text}")
                return self._fallback_classify(content)
            
            result = response.json()
            ai_answer = result['choices'][0]['message']['content'].strip()
            print(f"AI原始回答: {ai_answer}")
            
            matched_category = 'general'
            matched_label = '其他'
            confidence = 0.8
            
            for label, cat_id in self.categories.items():
                if label in ai_answer or ai_answer == label:
                    matched_category = cat_id
                    matched_label = label
                    confidence = 0.9
                    break
            
            if matched_category == 'general':
                fallback_result = self._fallback_classify(content)
                if fallback_result['confidence'] > confidence:
                    return fallback_result
            
            keywords = []
            for word in ['路灯', '道路', '农业', '补贴', '医保', '低保', '垃圾', '污水', '维修', '庄稼', '收成', '养殖']:
                if word in content:
                    keywords.append(word)
            
            return {
                'category': matched_category,
                'label': matched_label,
                'dept_name': self.dept_names.get(matched_category, '综合服务部'),
                'confidence': confidence,
                'need_manual': confidence < 0.7,
                'aiTags': keywords[:5]
            }
            
        except Exception as e:
            print(f"AI分类异常: {e}")
            return self._fallback_classify(content)
    
    def _fallback_classify(self, content):
        """后备分类方法（基于关键词）"""
        content_lower = content.lower()
        
        keyword_map = {
            'agriculture': {
                'keywords': ['农业', '补贴', '种植', '养殖', '农药', '化肥', '庄稼', '收成', '农田', '耕地', '粮食', '菜地', '果园', '水稻', '小麦', '玉米', '蔬菜', '大棚', '农技'],
                'label': '农业服务',
                'weight': 1.5
            },
            'infrastructure': {
                'keywords': ['路灯', '道路', '维修', '水管', '桥梁', '设施', '破损', '损坏', '施工', '修建', '水电', '网络', '信号', '公路', '村道', '水渠', '电线'],
                'label': '基建维修',
                'weight': 1.4
            },
            'environment': {
                'keywords': ['垃圾', '污水', '环境', '卫生', '污染', '清理', '清洁', '脏乱', '臭味', '绿化', '环保', '垃圾桶', '公厕', '河道', '排水'],
                'label': '环境卫生',
                'weight': 1.4
            },
            'health': {
                'keywords': ['医疗', '医保', '健康', '疫苗', '看病', '医院', '医生', '药品', '诊所', '卫生室', '住院', '报销', '体检', '养老', '卫生院'],
                'label': '医疗卫生',
                'weight': 1.3
            },
            'civil': {
                'keywords': ['低保', '救助', '婚姻', '户籍', '社保', '补贴', '申请', '证明', '登记', '福利', '养老', '残疾', '优抚', '五保', '扶贫'],
                'label': '民政咨询',
                'weight': 1.3
            }
        }
        
        scores = {}
        matched_keywords = []
        
        for cat_id, info in keyword_map.items():
            score = 0
            for keyword in info['keywords']:
                if keyword in content:
                    score += info['weight']
                    matched_keywords.append(keyword)
            scores[cat_id] = score
        
        max_score = 0
        best_category = 'general'
        best_label = '其他'
        
        for cat_id, score in scores.items():
            if score > max_score:
                max_score = score
                best_category = cat_id
                best_label = keyword_map[cat_id]['label']
        
        if max_score >= 3:
            confidence = 0.85
        elif max_score >= 2:
            confidence = 0.75
        elif max_score >= 1:
            confidence = 0.65
        else:
            confidence = 0.5
            best_category = 'general'
            best_label = '其他'
        
        unique_keywords = list(set(matched_keywords))[:5]
        
        return {
            'category': best_category,
            'label': best_label,
            'dept_name': self.dept_names.get(best_category, '综合服务部'),
            'confidence': confidence,
            'need_manual': confidence < 0.7,
            'aiTags': unique_keywords
        }
    
    def chat(self, question):
        """对话接口 - 使用Qwen3-14B模型"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """你是一个专业的乡村政策咨询助手，专门回答村民关于农业补贴、医保报销、低保申请、基建维修、环境卫生等方面的问题。
请用通俗易懂、亲切友好的语言回答，就像和村民面对面聊天一样。多使用口语化的表达，避免过于专业的术语。
如果遇到不知道的问题，就说"这个问题我需要查一下，您稍等"或者"建议您去村委会咨询"，不要编造信息。"""
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.7,
            "max_tokens": 500,
            "enable_thinking": False
        }
        
        try:
            print(f"正在调用魔搭API，模型: {self.model}")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
                verify=False
            )
            
            if response.status_code != 200:
                print(f"API调用失败: {response.status_code}")
                print(f"响应内容: {response.text}")
                return None, f"API调用失败: {response.status_code}"
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            return answer, None
            
        except requests.exceptions.Timeout:
            return None, "请求超时"
        except Exception as e:
            print(f"对话异常: {e}")
            return None, str(e)

# 创建全局AI服务实例
ai_service = AIService()

# ==================== 纯语义理解修正函数 ====================
def semantic_correction_with_llm(text):
    """使用魔搭社区Qwen模型进行纯语义理解修正"""
    if not text or len(text) < 2:
        return text, False
    
    system_prompt = """你是一个专业的乡村语音识别助手。请分析以下语音识别结果，基于上下文和常识进行修正。

任务要求：
1. 只返回修正后的文本，不要任何解释或额外内容
2. 如果语义通顺合理，直接返回原文本
3. 如果存在语义不通顺的地方，根据上下文推测最可能的正确文本
4. 保持原意不变，只修正明显的语音识别错误
5. 如果不确定如何修正，返回原文本

记住：你是纯粹基于语义理解来修正，不要依赖任何预设的关键词列表。"""

    prompt = f"""请分析以下语音识别结果，如果语义不通顺请修正：
识别结果：{text}
修正后："""

    headers = {
        "Authorization": f"Bearer {MODELSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "Qwen/Qwen3-14B",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 50,
        "enable_thinking": False
    }
    
    try:
        response = requests.post(
            f"{MODELSCOPE_BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            timeout=5,
            verify=False
        )
        
        if response.status_code == 200:
            result = response.json()
            corrected = result['choices'][0]['message']['content'].strip()
            
            need_confirm = False
            
            if corrected != text:
                print(f"   🤔 语义分析: 发现可能错误")
                print(f"     原文本: {text}")
                print(f"     修正后: {corrected}")
                
                if abs(len(corrected) - len(text)) > 5:
                    need_confirm = True
                    print(f"  ⚠️ 修正幅度较大，建议用户确认")
                
                return corrected, need_confirm
            
            return text, False
        else:
            print(f"LLM修正API调用失败: {response.status_code}")
            return text, False
            
    except Exception as e:
        print(f"LLM修正异常: {e}")
        return text, False

# ==================== 数据库模型定义 ====================
class User(db.Model):
    """用户模型"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    salt = db.Column(db.String(50), nullable=False)
    user_type = db.Column(db.Enum('villager', 'admin'), nullable=False, default='villager')
    real_name = db.Column(db.String(50), nullable=False)
    phone = db.Column(db.String(20), nullable=False, index=True)
    address = db.Column(db.String(200), nullable=False)
    avatar_url = db.Column(db.String(500))
    
    status = db.Column(db.Enum('pending', 'active', 'suspended', 'rejected'), 
                      nullable=False, default='pending')
    
    approved_at = db.Column(db.DateTime)
    reject_reason = db.Column(db.String(500))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, username, password, user_type='villager', real_name='', 
                 phone='', address='', avatar_url=None):
        self.username = username
        self.set_password(password)
        self.user_type = user_type
        self.real_name = real_name
        self.phone = phone
        self.address = address
        self.avatar_url = avatar_url
    
    def set_password(self, password):
        self.salt = secrets.token_hex(16)
        self.password_hash = generate_password_hash(f"{password}{self.salt}")
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, f"{password}{self.salt}")
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'user_type': self.user_type,
            'real_name': self.real_name,
            'phone': self.phone,
            'address': self.address,
            'avatar_url': self.avatar_url,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
        }
    
    def can_login(self):
        if self.status == 'pending':
            return False, '账号待审核，请等待管理员审核通过'
        elif self.status == 'rejected':
            return False, f'账号审核未通过: {self.reject_reason}'
        elif self.status == 'suspended':
            return False, '账号已停用，请联系管理员'
        elif self.status == 'active':
            return True, '可以登录'
        return False, '账号状态异常'


class Feedback(db.Model):
    """反馈模型"""
    __tablename__ = 'feedbacks'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50))
    status = db.Column(db.Enum('pending_ai', 'pending_manual', 'processing', 'completed', 'rejected', 'archived'),
                      default='pending_ai')
    ai_confidence = db.Column(db.Float)
    ai_result = db.Column(db.JSON)
    manual_category = db.Column(db.String(50))
    images = db.Column(db.JSON)
    
    voice_url = db.Column(db.String(500))
    voice_text = db.Column(db.Text)
    voice_duration = db.Column(db.Integer)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime)

    rating = db.Column(db.Integer, nullable=True)
    comment = db.Column(db.Text, nullable=True)
    evaluated_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship('User', foreign_keys=[user_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'content': self.content,
            'category': self.category,
            'status': self.status,
            'ai_confidence': self.ai_confidence,
            'manual_category': self.manual_category,
            'voice_url': self.voice_url,
            'voice_text': self.voice_text,
            'voice_duration': self.voice_duration,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Message(db.Model):
    """消息模型"""
    __tablename__ = 'messages'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    type = db.Column(db.Enum('user_audit', 'feedback_manual', 'feedback_result', 'system'),
                     nullable=False)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    related_id = db.Column(db.Integer)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', foreign_keys=[user_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'title': self.title,
            'content': self.content,
            'related_id': self.related_id,
            'is_read': self.is_read,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# ==================== 创建Flask应用 ====================
def create_app():
    # 设置时区为中国时区
    os.environ['TZ'] = 'Asia/Shanghai'
    try:
        time.tzset()
    except AttributeError:
        # Windows 系统不支持 tzset
        pass

    app = Flask(__name__)
    
    app.config['SECRET_KEY'] = 'township-service-secret-key-2024'
    app.config['JWT_SECRET_KEY'] = 'jwt-secret-key-township-2024'
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=2)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/township_service'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ECHO'] = True
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    db.init_app(app)
    jwt.init_app(app)
    
    with app.app_context():
        db.create_all()
        create_default_admin(app)
    
    register_routes(app)
    
    return app

# ==================== 创建默认管理员 ====================
def create_default_admin(app):
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin_user = User(
            username='admin',
            password='admin123',
            user_type='admin',
            real_name='系统管理员',
            phone='13800138000',
            address='乡镇政府办公室'
        )
        admin_user.status = 'active'
        admin_user.approved_at = datetime.utcnow()
        
        db.session.add(admin_user)
        db.session.commit()
        
        print("=" * 50)
        print("✅ 已创建默认管理员账号")
        print("   用户名: admin")
        print("   密码: admin123")
        print("=" * 50)

# ==================== 注册所有API路由 ====================
def register_routes(app):
    """注册所有API路由"""
    
    # ==================== 首页路由 ====================
    @app.route('/')
    def index():
        return jsonify({
            'code': 200,
            'message': '乡镇服务反馈平台后端服务',
            'data': {
                'name': '乡镇服务反馈平台API',
                'version': '1.0.0',
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'endpoints': {
                    'test': '/api/auth/test',
                    'register': '/api/auth/register',
                    'login': '/api/auth/login'
                }
            }
        })
    
    # ==================== 测试接口 ====================
    @app.route('/api/auth/test', methods=['GET'])
    def test_connection():
        try:
            db.session.execute('SELECT 1').scalar()
            db_status = 'connected'
        except Exception as e:
            db_status = f'disconnected: {str(e)}'
        
        return jsonify({
            'code': 200,
            'message': '后端API连接正常',
            'data': {
                'service': '乡镇服务反馈平台',
                'api_version': '1.0.0',
                'database': db_status,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    # ==================== 用户注册接口 ====================
    @app.route('/api/auth/register', methods=['POST'])
    def register():
        try:
            data = request.get_json()
            print("\n📝 收到注册请求:", data)
            
            required_fields = ['username', 'password', 'real_name', 'phone', 'address']
            for field in required_fields:
                if field not in data or not data[field]:
                    return jsonify({'code': 400, 'message': f'{field}不能为空', 'data': None}), 400
            
            if len(data['username']) < 3 or len(data['username']) > 20:
                return jsonify({'code': 400, 'message': '用户名长度应为3-20位', 'data': None}), 400
            
            if len(data['password']) < 6 or len(data['password']) > 20:
                return jsonify({'code': 400, 'message': '密码长度应为6-20位', 'data': None}), 400
            
            if not re.match(r'^1[3-9]\d{9}$', data['phone']):
                return jsonify({'code': 400, 'message': '请输入正确的手机号', 'data': None}), 400

            # 处理用户名冲突：如果原账户已被拒绝，则清理旧记录允许重新注册
            existing_user = User.query.filter_by(username=data['username']).first()
            if existing_user:
                if existing_user.status == 'rejected':
                    try:
                        db.session.delete(existing_user)
                        db.session.flush()
                        print(f"🧹 已清理被拒绝用户 {existing_user.username}，允许重新注册")
                    except Exception as e:
                        print(f"删除旧用户失败: {e}")
                        db.session.rollback()
                        return jsonify({'code': 500, 'message': '系统错误，请稍后重试', 'data': None}), 500
                else:
                    return jsonify({'code': 400, 'message': '用户名已存在', 'data': None}), 400
            
            # 处理手机号冲突
            existing_phone = User.query.filter_by(phone=data['phone']).first()
            if existing_phone:
                if existing_phone.status == 'rejected':
                    try:
                        # 如果手机号对应的用户和上面刚删的不是同一个人，也需要清理
                        if existing_user is None or existing_phone.id != existing_user.id:
                            db.session.delete(existing_phone)
                            db.session.flush()
                            print(f"🧹 已清理被拒绝用户 {existing_phone.username}（手机号冲突），允许重新注册")
                    except Exception as e:
                        print(f"删除旧用户失败: {e}")
                        db.session.rollback()
                        return jsonify({'code': 500, 'message': '系统错误，请稍后重试', 'data': None}), 500
                else:
                    return jsonify({'code': 400, 'message': '该手机号已注册', 'data': None}), 400
            
            user = User(
                username=data['username'],
                password=data['password'],
                user_type='villager',
                real_name=data['real_name'],
                phone=data['phone'],
                address=data['address']
            )
            
            db.session.add(user)
            db.session.flush()
            
            try:
                admins = User.query.filter_by(user_type='admin', status='active').all()
                for admin in admins:
                    msg = Message(
                        user_id=admin.id,
                        type='user_audit',
                        title='新用户注册待审核',
                        content=f'用户"{data["real_name"]}"（{data["username"]}）申请注册，请及时审核',
                        related_id=user.id
                    )
                    db.session.add(msg)
                print(f"✅ 已为 {len(admins)} 个管理员创建消息")
            except Exception as msg_error:
                print(f"⚠️ 创建管理员消息失败: {str(msg_error)}")
            
            db.session.commit()
            print(f"✅ 用户注册成功并已提交到数据库: {user.username}")
            
            return jsonify({
                'code': 200,
                'message': '注册成功，请等待管理员审核',
                'data': {
                    'user_id': user.id,
                    'username': user.username,
                    'real_name': user.real_name,
                    'status': user.status
                }
            }), 200
            
        except Exception as e:
            print(f"❌ 注册异常: {str(e)}")
            db.session.rollback()
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500
    
    # ==================== 用户登录接口 ====================
    @app.route('/api/auth/login', methods=['POST'])
    def login():
        try:
            data = request.get_json()
            print("\n🔑 收到登录请求:", data)
            
            if 'username' not in data or not data['username']:
                return jsonify({'code': 400, 'message': '请输入用户名', 'data': None}), 400
            
            if 'password' not in data or not data['password']:
                return jsonify({'code': 400, 'message': '请输入密码', 'data': None}), 400
            
            user = User.query.filter_by(username=data['username']).first()
            if not user:
                return jsonify({'code': 401, 'message': '用户名或密码错误', 'data': None}), 401
            
            if not user.check_password(data['password']):
                return jsonify({'code': 401, 'message': '用户名或密码错误', 'data': None}), 401
            
            can_login, msg = user.can_login()
            if not can_login:
                return jsonify({'code': 403, 'message': msg, 'data': None}), 403
            
            db.session.commit()
            
            access_token = create_access_token(
                identity=str(user.id),
                additional_claims={
                    'username': user.username,
                    'user_type': user.user_type,
                    'real_name': user.real_name
                }
            )
            
            print(f"✅ 登录成功: {user.username}")
            
            return jsonify({
                'code': 200,
                'message': '登录成功',
                'data': {
                    'access_token': access_token,
                    'user': user.to_dict()
                }
            }), 200
            
        except Exception as e:
            print(f"❌ 登录异常: {str(e)}")
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500
    
    # ==================== 获取当前用户信息接口 ====================
    @app.route('/api/auth/me', methods=['GET'])
    @jwt_required()
    def get_current_user():
        try:
            current_user_id = get_jwt_identity()
            user = User.query.get(current_user_id)
            
            if not user:
                return jsonify({'code': 404, 'message': '用户不存在', 'data': None}), 404
            
            return jsonify({
                'code': 200,
                'message': '获取成功',
                'data': user.to_dict()
            }), 200
            
        except Exception as e:
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 修改密码接口 ====================
    @app.route('/api/user/changePassword', methods=['POST'])
    @jwt_required()
    def change_password():
        try:
            current_user_id = get_jwt_identity()
            data = request.get_json()
            
            print(f"🔑 收到修改密码请求: 用户 {current_user_id}")
            
            old_password = data.get('oldPassword')
            new_password = data.get('newPassword')
            
            if not old_password:
                return jsonify({'code': 400, 'message': '请输入原密码', 'data': None}), 400
            
            if not new_password:
                return jsonify({'code': 400, 'message': '请输入新密码', 'data': None}), 400
            
            if len(new_password) < 6 or len(new_password) > 20:
                return jsonify({'code': 400, 'message': '新密码长度应为6-20位', 'data': None}), 400
            
            user = User.query.get(int(current_user_id))
            if not user:
                return jsonify({'code': 404, 'message': '用户不存在', 'data': None}), 404
            
            if not user.check_password(old_password):
                return jsonify({'code': 401, 'message': '原密码错误', 'data': None}), 401
            
            user.set_password(new_password)
            db.session.commit()
            
            print(f"✅ 密码修改成功: 用户 {user.username}")
            
            return jsonify({
                'code': 200,
                'message': '密码修改成功',
                'data': None
            }), 200
            
        except Exception as e:
            print(f"❌ 修改密码异常: {str(e)}")
            db.session.rollback()
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 更新用户信息接口 ====================
    @app.route('/api/user/update', methods=['POST'])
    @jwt_required()
    def update_user_info():
        try:
            current_user_id = get_jwt_identity()
            data = request.get_json()
            
            print(f"📝 收到更新用户信息请求: 用户 {current_user_id}, 数据 {data}")
            
            user = User.query.get(int(current_user_id))
            if not user:
                return jsonify({'code': 404, 'message': '用户不存在', 'data': None}), 404
            
            if data.get('nickname'):
                user.nickname = data.get('nickname')
            
            if data.get('name') or data.get('real_name'):
                user.real_name = data.get('name') or data.get('real_name') or user.real_name
            
            if data.get('village') or data.get('region'):
                user.address = data.get('village') or data.get('region') or user.address
            
            if data.get('avatar'):
                user.avatar_url = data.get('avatar')
            
            db.session.commit()
            
            print(f"✅ 用户信息更新成功: {user.username}")
            
            return jsonify({
                'code': 200,
                'message': '更新成功',
                'data': user.to_dict()
            }), 200
            
        except Exception as e:
            print(f"❌ 更新用户信息异常: {str(e)}")
            db.session.rollback()
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 获取消息列表接口 ====================
    @app.route('/api/messages', methods=['GET'])
    @jwt_required()
    def get_messages():
        try:
            current_user_id = get_jwt_identity()
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 20, type=int)
            unread_only = request.args.get('unread_only', 'false').lower() == 'true'
            
            query = Message.query.filter_by(user_id=current_user_id)
            if unread_only:
                query = query.filter_by(is_read=False)
            
            pagination = query.order_by(Message.created_at.desc()).paginate(
                page=page, per_page=per_page, error_out=False
            )
            
            messages = [msg.to_dict() for msg in pagination.items]
            unread_count = Message.query.filter_by(user_id=current_user_id, is_read=False).count()
            
            return jsonify({
                'code': 200,
                'message': '获取成功',
                'data': {
                    'messages': messages,
                    'unread_count': unread_count,
                    'pagination': {
                        'page': pagination.page,
                        'per_page': pagination.per_page,
                        'total': pagination.total,
                        'pages': pagination.pages
                    }
                }
            }), 200
            
        except Exception as e:
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500
    
    # ==================== 标记消息已读接口 ====================
    @app.route('/api/messages/<int:message_id>/read', methods=['PUT'])
    @jwt_required()
    def mark_message_read(message_id):
        try:
            current_user_id = get_jwt_identity()
            message = Message.query.filter_by(id=message_id, user_id=current_user_id).first()
            
            if not message:
                return jsonify({'code': 404, 'message': '消息不存在', 'data': None}), 404
            
            message.is_read = True
            db.session.commit()
            
            return jsonify({
                'code': 200,
                'message': '已标记为已读',
                'data': message.to_dict()
            }), 200
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500
    
    # ==================== 全部标记已读接口 ====================
    @app.route('/api/messages/read-all', methods=['PUT'])
    @jwt_required()
    def mark_all_read():
        try:
            current_user_id = get_jwt_identity()
            
            messages = Message.query.filter_by(user_id=current_user_id, is_read=False).all()
            now = datetime.utcnow()
            
            for msg in messages:
                msg.is_read = True
            
            db.session.commit()
            
            return jsonify({
                'code': 200,
                'message': '全部已标记为已读',
                'data': {
                    'count': len(messages)
                }
            }), 200
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500
    
    # ==================== 删除消息接口 ====================
    @app.route('/api/messages/<int:message_id>/delete', methods=['DELETE'])
    @jwt_required()
    def delete_message(message_id):
        try:
            current_user_id = get_jwt_identity()
            message = Message.query.filter_by(id=message_id, user_id=current_user_id).first()
            
            if not message:
                return jsonify({'code': 404, 'message': '消息不存在', 'data': None}), 404
            
            db.session.delete(message)
            db.session.commit()
            
            return jsonify({
                'code': 200,
                'message': '消息已删除',
                'data': {'id': message_id}
            }), 200
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500
    
    # ==================== 管理员接口 - 获取待审核用户列表 ====================
    @app.route('/api/admin/users/pending', methods=['GET'])
    @jwt_required()
    def get_pending_users():
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.get(current_user_id)
            
            if not current_user or current_user.user_type != 'admin':
                return jsonify({'code': 403, 'message': '权限不足', 'data': None}), 403
            
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 20, type=int)
            
            pagination = User.query.filter_by(status='pending').order_by(
                User.created_at.desc()
            ).paginate(page=page, per_page=per_page, error_out=False)
            
            users = [{
                'id': user.id,
                'username': user.username,
                'real_name': user.real_name,
                'phone': user.phone,
                'address': user.address,
                'created_at': user.created_at.isoformat() if user.created_at else None
            } for user in pagination.items]
            
            return jsonify({
                'code': 200,
                'message': '获取成功',
                'data': {
                    'users': users,
                    'pagination': {
                        'page': pagination.page,
                        'per_page': pagination.per_page,
                        'total': pagination.total,
                        'pages': pagination.pages
                    }
                }
            }), 200
            
        except Exception as e:
            print(f"获取待审核用户异常: {str(e)}")
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500
    
    # ==================== 管理员接口 - 通过用户审核 ====================
    @app.route('/api/admin/users/<int:user_id>/approve', methods=['POST'])
    @jwt_required()
    def approve_user(user_id):
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.get(current_user_id)
            
            if not current_user or current_user.user_type != 'admin':
                return jsonify({'code': 403, 'message': '权限不足', 'data': None}), 403
            
            user = User.query.get(user_id)
            if not user:
                return jsonify({'code': 404, 'message': '用户不存在', 'data': None}), 404
            
            if user.status != 'pending':
                return jsonify({'code': 400, 'message': f'用户当前状态为{user.status}，无需审核', 'data': None}), 400
            
            user.status = 'active'
            user.approved_at = datetime.utcnow()
            
            msg = Message(
                user_id=user.id,
                type='system',
                title='账号审核通过',
                content='您的注册申请已通过审核，现在可以登录使用平台了！',
                related_id=user.id
            )
            db.session.add(msg)
            db.session.commit()
            
            return jsonify({'code': 200, 'message': '审核通过成功', 'data': user.to_dict()}), 200
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500
    
    # ==================== 管理员接口 - 拒绝用户审核 ====================
    @app.route('/api/admin/users/<int:user_id>/reject', methods=['POST'])
    @jwt_required()
    def reject_user(user_id):
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.get(current_user_id)
            
            if not current_user or current_user.user_type != 'admin':
                return jsonify({'code': 403, 'message': '权限不足', 'data': None}), 403
            
            data = request.get_json()
            raw_reason = data.get('reason', '')
            
            # 防御性处理：如果原因为空，或者包含了弹窗的确认文案，则用默认原因
            safe_reason = '未通过审核'
            if raw_reason and raw_reason.strip():
                # 如果原因里出现了典型的弹窗提示文字，说明是误传，替换掉
                if '确定要拒绝' in raw_reason or '注册申请吗' in raw_reason:
                    safe_reason = '未通过审核'
                else:
                    safe_reason = raw_reason.strip()
            
            user = User.query.get(user_id)
            if not user:
                return jsonify({'code': 404, 'message': '用户不存在', 'data': None}), 404
            
            if user.status != 'pending':
                return jsonify({'code': 400, 'message': f'用户当前状态为{user.status}，无法拒绝', 'data': None}), 400
            
            user.status = 'rejected'
            user.reject_reason = safe_reason
            user.approved_at = datetime.utcnow()
            
            db.session.commit()
            
            return jsonify({
                'code': 200,
                'message': '已拒绝该用户',
                'data': {'user_id': user.id, 'reason': safe_reason}
            }), 200
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500
    
    # ==================== 管理员接口 - 获取待人工分类反馈列表 ====================
    @app.route('/api/admin/feedbacks/pending-manual', methods=['GET'])
    @jwt_required()
    def get_pending_feedbacks():
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.get(current_user_id)
            
            if not current_user or current_user.user_type != 'admin':
                return jsonify({'code': 403, 'message': '权限不足', 'data': None}), 403
            
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 20, type=int)
            keyword = request.args.get('keyword', '')
            sort = request.args.get('sort', 'time_desc')
            
            print(f"📊 排序参数: {sort}, 页码: {page}, 每页: {per_page}")
            
            # 构建基础查询
            query = Feedback.query.filter(
                Feedback.status.in_(['pending_manual', 'processing'])
            )
            
            # 如果有关键词，添加搜索条件
            if keyword and keyword.strip():
                search_term = f"%{keyword.strip()}%"
                query = query.filter(
                    db.or_(
                        Feedback.title.ilike(search_term),
                        Feedback.content.ilike(search_term)
                    )
                )
                print(f"🔍 搜索关键词: {keyword}")
            
            # 排序
            if sort == 'time_asc':
                query = query.order_by(Feedback.created_at.asc())
                print("📅 排序方式: 最早优先 (升序)")
            else:
                query = query.order_by(Feedback.created_at.desc())
                print("📅 排序方式: 最新优先 (降序)")
            
            # 分页
            pagination = query.paginate(page=page, per_page=per_page, error_out=False)
            
            print(f"📋 查询到 {pagination.total} 条总记录，当前页 {len(pagination.items)} 条")
            
            feedbacks = []
            for fb in pagination.items:
                user = User.query.get(fb.user_id)
                
                # 【修复】处理图片字段 - 兼容多种格式
                images = []
                if fb.images:
                    if isinstance(fb.images, list):
                        images = fb.images
                    elif isinstance(fb.images, str):
                        try:
                            # 尝试解析JSON字符串
                            images = json.loads(fb.images)
                            if not isinstance(images, list):
                                images = [fb.images]
                        except:
                            # 如果不是JSON，直接作为单个图片
                            images = [fb.images]
                
                print(f"🖼️ 反馈 ID {fb.id} 图片数量: {len(images)}")
                
                feedbacks.append({
                    'id': fb.id,
                    'title': fb.title,
                    'content': fb.content,
                    'status': fb.status,
                    'ai_confidence': fb.ai_confidence,
                    'ai_tags': fb.ai_result.get('aiTags', []) if fb.ai_result else [],
                    'user': {
                        'id': user.id,
                        'real_name': user.real_name,
                        'phone': user.phone,
                        'address': user.address
                    } if user else None,
                    'voice_url': fb.voice_url,
                    'voice_text': fb.voice_text,
                    'voice_duration': fb.voice_duration,
                    'created_at': fb.created_at.isoformat() if fb.created_at else None,
                    'images': images  # 返回处理后的图片列表
                })
            
            return jsonify({
                'code': 200,
                'message': '获取成功',
                'data': {
                    'feedbacks': feedbacks,
                    'pagination': {
                        'page': pagination.page,
                        'per_page': pagination.per_page,
                        'total': pagination.total,
                        'pages': pagination.pages
                    }
                }
            }), 200
            
        except Exception as e:
            print(f"❌ 获取待分类反馈异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500

    # ==================== 管理员接口 - 人工分类反馈 ====================
    @app.route('/api/admin/feedbacks/<int:feedback_id>/classify', methods=['POST'])
    @jwt_required()
    def classify_feedback(feedback_id):
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.get(current_user_id)
            
            if not current_user or current_user.user_type != 'admin':
                return jsonify({'code': 403, 'message': '权限不足', 'data': None}), 403
            
            data = request.get_json()
            category = data.get('category')
            opinion = data.get('opinion', '')
            
            if not category:
                return jsonify({'code': 400, 'message': '请选择分类', 'data': None}), 400
            
            feedback = Feedback.query.get(feedback_id)
            if not feedback:
                return jsonify({'code': 404, 'message': '反馈不存在', 'data': None}), 404
            
            if feedback.status not in ['pending_manual', 'processing']:
                return jsonify({
                    'code': 400,
                    'message': f'该反馈当前状态为{feedback.status}，无法进行分类',
                    'data': None
                }), 400

            old_status = feedback.status
            old_category = feedback.category

            feedback.manual_category = category
            feedback.category = category
            feedback.status = 'processing'
            feedback.processed_at = datetime.utcnow()

            print(f"管理员 {current_user.username} 将反馈 {feedback_id} 从 {old_status} 人工分类到 {category}")

            dept_names = {
                'agriculture': '农业服务部',
                'infrastructure': '基建维修部',
                'environment': '环境整治部',
                'health': '医疗卫生部',
                'civil': '民政服务部',
                'general': '综合服务部'
            }
            dept_display = dept_names.get(category, category)

            msg = Message(
                user_id=feedback.user_id,
                type='feedback_result',
                title='反馈已分类处理',
                content=f'您的反馈"{feedback.title}"已被归类为"{dept_display}"，相关部门正在处理中',
                related_id=feedback.id
            )
            db.session.add(msg)

            db.session.commit()

            return jsonify({
                'code': 200,
                'message': '分类成功',
                'data': {
                    'feedback_id': feedback.id,
                    'category': category,
                    'status': feedback.status,
                    'ai_category': old_category
                }
            }), 200

        except Exception as e:
            print(f"分类反馈异常: {str(e)}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 管理员接口 - 标记反馈为已处理 ====================
    @app.route('/api/admin/feedbacks/<int:feedback_id>/complete', methods=['POST'])
    @jwt_required()
    def complete_feedback(feedback_id):
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.get(current_user_id)
            
            if not current_user or current_user.user_type != 'admin':
                return jsonify({'code': 403, 'message': '权限不足', 'data': None}), 403
            
            feedback = Feedback.query.get(feedback_id)
            if not feedback:
                return jsonify({'code': 404, 'message': '反馈不存在', 'data': None}), 404
            
            if feedback.status not in ['pending_manual', 'processing']:
                return jsonify({
                    'code': 400, 
                    'message': f'当前状态为{feedback.status}，无法标记为已处理', 
                    'data': None
                }), 400
            
            old_status = feedback.status
            
            if not feedback.manual_category and feedback.category:
                feedback.manual_category = feedback.category
            
            feedback.status = 'completed'
            feedback.processed_at = datetime.utcnow()
            
            print(f"管理员 {current_user.username} 将反馈 {feedback_id} 从 {old_status} 标记为已完成")
            
            msg = Message(
                user_id=feedback.user_id,
                type='feedback_result',
                title='反馈已处理完成',
                content=f'您的反馈"{feedback.title}"已被处理完成，感谢您的反馈！',
                related_id=feedback.id
            )
            db.session.add(msg)
            
            db.session.commit()
            
            return jsonify({
                'code': 200,
                'message': '已标记为已处理',
                'data': {
                    'feedback_id': feedback.id,
                    'status': feedback.status,
                    'department': feedback.manual_category or feedback.category,
                    'processed_at': feedback.processed_at.isoformat() if feedback.processed_at else None
                }
            }), 200
            
        except Exception as e:
            print(f"标记反馈完成异常: {str(e)}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 用户评价反馈接口 ====================
    @app.route('/api/feedback/<int:feedback_id>/evaluate', methods=['POST'])
    @jwt_required()
    def evaluate_feedback(feedback_id):
        try:
            current_user_id = get_jwt_identity()
            data = request.get_json()
        
            rating = data.get('rating')
            comment = data.get('comment', '')
        
            if not rating:
                return jsonify({'code': 400, 'message': '请给出评分', 'data': None}), 400
        
            if rating < 1 or rating > 5:
                return jsonify({'code': 400, 'message': '评分必须为1-5分', 'data': None}), 400
        
            feedback = Feedback.query.get(feedback_id)
            if not feedback:
                return jsonify({'code': 404, 'message': '反馈不存在', 'data': None}), 404
        
            if feedback.user_id != int(current_user_id):
                return jsonify({'code': 403, 'message': '无权评价此反馈', 'data': None}), 403
        
            if feedback.status != 'completed':
                return jsonify({'code': 400, 'message': '只有已完成的反馈才能评价', 'data': None}), 400
        
            if feedback.rating is not None:
                return jsonify({'code': 400, 'message': '该反馈已经评价过了', 'data': None}), 400
        
            feedback.rating = rating
            feedback.comment = comment
            feedback.evaluated_at = datetime.utcnow()
        
            feedback.status = 'archived'
         
            db.session.commit()
        
            print(f"✅ 用户 {current_user_id} 对反馈 {feedback_id} 进行了评价")
            print(f"   评分: {rating} 星")
            print(f"   评价内容: {comment}")
            print(f"   反馈状态已更新为: archived (已归档)")
        
            return jsonify({
                'code': 200,
                'message': '评价成功',
                'data': {
                    'feedback_id': feedback.id,
                    'rating': rating,
                    'comment': comment,
                    'status': 'archived',
                    'status_display': '已归档',
                    'evaluated_at': feedback.evaluated_at.isoformat() if feedback.evaluated_at else None
                }
            }), 200
        
        except Exception as e:
            print(f"❌ 评价反馈异常: {str(e)}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 管理员统计接口 ====================
    @app.route('/api/admin/statistics', methods=['GET'])
    @jwt_required()
    def get_admin_statistics():
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.get(current_user_id)

            if not current_user or current_user.user_type != 'admin':
                return jsonify({'code': 403, 'message': '权限不足', 'data': None}), 403

            pending_users = User.query.filter_by(status='pending').count()
            pending_feedbacks = Feedback.query.filter_by(status='pending_manual').count()

            return jsonify({
                'code': 200,
                'message': '获取成功',
                'data': {
                    'pending_users': pending_users,
                    'pending_feedbacks': pending_feedbacks
                }
            }), 200

        except Exception as e:
            print(f"获取管理员统计异常: {str(e)}")
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500

    # ==================== 反馈统计接口 ====================
    @app.route('/api/feedback/stat', methods=['GET'])
    @jwt_required()
    def get_feedback_stat():
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.get(current_user_id)
            
            if not current_user:
                return jsonify({'code': 404, 'message': '用户不存在', 'data': None}), 404
            
            if current_user.user_type == 'admin':
                # ==================== 管理员视角：统计所有反馈 ====================
                
                # 1. 总反馈数（所有反馈）
                total = Feedback.query.count()
                
                # 2. 待处理（pending_manual + processing）
                pending_manual = Feedback.query.filter_by(status='pending_manual').count()
                processing = Feedback.query.filter_by(status='processing').count()
                pending_total = pending_manual + processing
                
                # 3. 待评价（completed 状态）
                pending_evaluate = Feedback.query.filter_by(status='completed').count()
                
                # 4. 已评价（archived 状态）
                evaluated = Feedback.query.filter_by(status='archived').count()
                
                # 5. 已完成总数（待评价 + 已评价）
                completed_total = pending_evaluate + evaluated
                
                # 6. 满意度计算：基于所有已评价且有评分的反馈
                # 查询所有 archived 状态且有评分的反馈
                evaluated_feedbacks = Feedback.query.filter(
                    Feedback.status == 'archived',
                    Feedback.rating.isnot(None)
                ).all()
                
                if evaluated_feedbacks:
                    total_rating = sum(fb.rating for fb in evaluated_feedbacks)
                    avg_rating = total_rating / len(evaluated_feedbacks)
                    # 5星为100%，4星为80%，3星为60%，2星为40%，1星为20%
                    satisfaction_rate = int((avg_rating / 5) * 100)
                else:
                    satisfaction_rate = 0
                
                print(f"📊 管理员统计: 总数={total}, 待处理={pending_total}, "
                      f"待评价={pending_evaluate}, 已评价={evaluated}, 满意度={satisfaction_rate}%")
                
                return jsonify({
                    'code': 200,
                    'message': '获取成功',
                    'data': {
                        'total': total,                       # 总反馈数
                        'pending_total': pending_total,       # 待处理总数（待分类+处理中）
                        'pending_manual': pending_manual,     # 待分类
                        'processing': processing,             # 处理中
                        'pending_evaluate': pending_evaluate, # 待评价
                        'evaluated': evaluated,               # 已评价
                        'completed_total': completed_total,   # 已完成总数（用于进度条）
                        'satisfactionRate': satisfaction_rate # 满意度百分比
                    }
                }), 200
                
            else:
                # ==================== 村民视角：只统计当前用户的反馈 ====================
                total = Feedback.query.filter_by(user_id=current_user_id).count()
                processing = Feedback.query.filter_by(user_id=current_user_id, status='processing').count()
                completed = Feedback.query.filter(
                    Feedback.user_id == current_user_id,
                    Feedback.status.in_(['completed', 'archived'])
                ).count()
                
                # 村民的满意度（只计算该用户自己的评价）
                evaluated = Feedback.query.filter(
                    Feedback.user_id == current_user_id,
                    Feedback.rating.isnot(None)
                ).all()
                if evaluated:
                    total_rating = sum(fb.rating for fb in evaluated)
                    avg_rating = total_rating / len(evaluated)
                    satisfaction_rate = int((avg_rating / 5) * 100)
                else:
                    satisfaction_rate = 0
                
                return jsonify({
                    'code': 200,
                    'message': '获取成功',
                    'data': {
                        'total': total,
                        'processing': processing,
                        'completed': completed,
                        'satisfactionRate': satisfaction_rate
                    }
                }), 200
                
        except Exception as e:
            print(f"❌ 获取反馈统计异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500

    # ==================== 获取用户反馈列表接口 ====================
    @app.route('/api/feedback/list', methods=['GET'])
    @jwt_required()
    def get_feedback_list():
        try:
            current_user_id = get_jwt_identity()
            print(f"获取反馈列表，用户ID: {current_user_id}")

            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 10, type=int)
            status = request.args.get('status', '')

            current_user = User.query.get(int(current_user_id))
            if not current_user:
                return jsonify({'code': 404, 'message': '用户不存在', 'data': None}), 404

            query = Feedback.query.filter_by(user_id=int(current_user_id))

            # 处理状态筛选
            if status and status.strip():
                if status == 'archived':
                    # 已评分状态：返回状态为 archived 或者有评分的反馈
                    query = query.filter(
                        db.or_(
                            Feedback.status == 'archived',
                            Feedback.rating.isnot(None)
                        )
                    )
                elif status == 'completed':
                    # 已完成包括 completed 和 archived
                    query = query.filter(Feedback.status.in_(['completed', 'archived']))
                elif status == 'processing':
                    query = query.filter_by(status='processing')
                elif status == 'pending_ai':
                    query = query.filter_by(status='pending_ai')
                elif status == 'pending_manual':
                    query = query.filter_by(status='pending_manual')
                elif status == 'rejected':
                    query = query.filter_by(status='rejected')
                else:
                    query = query.filter_by(status=status)

            query = query.order_by(Feedback.created_at.desc())

            pagination = query.paginate(page=page, per_page=per_page, error_out=False)

            feedbacks = []
            for fb in pagination.items:
                status_display = '待分类'
                if fb.status == 'processing':
                    status_display = '处理中'
                elif fb.status == 'pending_ai':
                    status_display = 'AI处理中'
                elif fb.status == 'completed':
                    status_display = '已完成'
                elif fb.status == 'rejected':
                    status_display = '已驳回'
                elif fb.status == 'archived':
                    status_display = '已评分'  # 修改显示文本为"已评分"

                # 如果有评分但状态不是 archived，也显示为已评分
                if fb.rating is not None and fb.status != 'archived':
                    status_display = '已评分'

                dept_names = {
                    'agriculture': '农业服务部',
                    'infrastructure': '基建维修部',
                    'environment': '环境整治部',
                    'health': '医疗卫生部',
                    'civil': '民政服务部',
                    'general': '综合服务部'
                }

                dept_name = '待派单'
                department_id = fb.manual_category or fb.category
                
                if department_id:
                    if department_id in dept_names.values():
                        dept_name = department_id
                    else:
                        dept_name = dept_names.get(department_id, '待派单')

                images = []
                if fb.images:
                    if isinstance(fb.images, list):
                        images = fb.images
                    elif isinstance(fb.images, str):
                        try:
                            images = json.loads(fb.images)
                        except:
                            images = []

                # 判断是否有评价
                has_evaluated = fb.rating is not None

                # 如果有评分，强制 has_evaluated 为 True
                if fb.rating is not None:
                    has_evaluated = True

                feedbacks.append({
                    'id': fb.id,
                    'title': fb.title or '无标题',
                    'content': fb.content or '',
                    'category': fb.category,
                    'status': fb.status or 'pending_manual',
                    'status_display': status_display,
                    'dept_name': dept_name,
                    'dept_id': department_id,
                    'ai_confidence': fb.ai_confidence or 0,
                    'created_at': fb.created_at.isoformat() if fb.created_at else None,
                    'images': images,
                    'voice_url': fb.voice_url or '',
                    'voice_text': fb.voice_text or '',
                    'voice_duration': fb.voice_duration or 0,
                    'manual_category': fb.manual_category or '',
                    'has_evaluated': has_evaluated,
                    'rating': fb.rating or 0,
                    'comment': fb.comment or '',
                    'evaluated_at': fb.evaluated_at.isoformat() if fb.evaluated_at else None 
                })

            response_data = {
                'code': 200,
                'message': '获取成功',
                'data': {
                    'feedbacks': feedbacks,
                    'pagination': {
                        'page': pagination.page,
                        'per_page': pagination.per_page,
                        'total': pagination.total,
                        'pages': pagination.pages
                    }
                }
            }

            print(f"✅ 获取反馈列表成功，共 {len(feedbacks)} 条")
            return jsonify(response_data), 200

        except Exception as e:
            print(f"❌ 获取反馈列表异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'code': 500,
                'message': f'服务器内部错误: {str(e)}',
                'data': None
            }), 500

    # ==================== 获取反馈详情接口 ====================
    @app.route('/api/feedback/detail/<int:feedback_id>', methods=['GET'])
    @jwt_required()
    def get_feedback_detail(feedback_id):
        try:
            current_user_id = get_jwt_identity()
            feedback = Feedback.query.filter_by(id=feedback_id).first()
            if not feedback:
                return jsonify({'code': 404, 'message': '反馈不存在', 'data': None}), 404
            current_user = User.query.get(int(current_user_id))
            if not current_user:
                return jsonify({'code': 404, 'message': '用户不存在', 'data': None}), 404
            if feedback.user_id != int(current_user_id) and current_user.user_type != 'admin':
                return jsonify({'code': 403, 'message': '无权查看此反馈', 'data': None}), 403
            user = User.query.get(feedback.user_id)
            status_display_map = {
                'pending_ai': 'AI处理中',
                'pending_manual': '待分类',
                'processing': '处理中',
                'completed': '已完成',
                'rejected': '已驳回',
                'archived': '已评分'
            }
            dept_names = {
                'agriculture': '农业服务部',
                'infrastructure': '基建维修部',
                'environment': '环境整治部',
                'health': '医疗卫生部',
                'civil': '民政服务部',
                'general': '综合服务部'
            }
            dept_name = '待派单'
            ai_tags = []
            department_id = feedback.manual_category or feedback.category
            if department_id:
                if department_id in dept_names.values():
                    dept_name = department_id
                else:
                    dept_name = dept_names.get(department_id, '综合服务部')
            elif feedback.ai_result:
                if isinstance(feedback.ai_result, dict):
                    ai_tags = feedback.ai_result.get('aiTags', [])
                    ai_label = feedback.ai_result.get('label', '')
                    if ai_label:
                        dept_name = f"AI推荐:{ai_label}"
                elif isinstance(feedback.ai_result, list):
                    ai_tags = feedback.ai_result
                    if ai_tags and len(ai_tags) > 0:
                        dept_name = f"AI推荐:{ai_tags[0]}"
                elif isinstance(feedback.ai_result, str):
                    try:
                        ai_data = json.loads(feedback.ai_result)
                        if isinstance(ai_data, dict):
                            ai_tags = ai_data.get('aiTags', [])
                            ai_label = ai_data.get('label', '')
                            if ai_label:
                                dept_name = f"AI推荐:{ai_label}"
                        elif isinstance(ai_data, list):
                            ai_tags = ai_data
                            if ai_tags and len(ai_tags) > 0:
                                dept_name = f"AI推荐:{ai_tags[0]}"
                    except:
                        ai_tags = [feedback.ai_result]
                        dept_name = f"AI推荐:{feedback.ai_result[:10]}"
            images = []
            if feedback.images:
                if isinstance(feedback.images, list):
                    images = feedback.images
                elif isinstance(feedback.images, str):
                    try:
                        images = json.loads(feedback.images)
                    except:
                        images = []
            detail = {
                'id': feedback.id,
                'title': feedback.title,
                'content': feedback.content,
                'category': feedback.category,
                'status': feedback.status,
                'status_display': status_display_map.get(feedback.status, feedback.status),
                'dept_name': dept_name,
                'dept_id': department_id,
                'ai_confidence': feedback.ai_confidence,
                'ai_result': feedback.ai_result,
                'ai_tags': ai_tags,
                'manual_category': feedback.manual_category,
                'images': images,
                'voice_url': feedback.voice_url or '',
                'voice_text': feedback.voice_text or '',
                'voice_duration': feedback.voice_duration or 0,
                'created_at': feedback.created_at.isoformat() if feedback.created_at else None,
                'processed_at': feedback.processed_at.isoformat() if feedback.processed_at else None,
                'rating': feedback.rating,
                'comment': feedback.comment,
                'evaluated_at': feedback.evaluated_at.isoformat() if feedback.evaluated_at else None,
                'user': {
                    'id': user.id,
                    'real_name': user.real_name,
                    'phone': user.phone,
                    'address': user.address
                } if user else None
            }
            print(f"✅ 获取反馈详情成功, ID: {feedback_id}, 处理部门: {dept_name}")
            return jsonify({'code': 200, 'message': '获取成功', 'data': detail}), 200
        except Exception as e:
            print(f"❌ 获取反馈详情异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 删除反馈接口 ====================
    @app.route('/api/feedback/<int:feedback_id>/delete', methods=['DELETE'])
    @jwt_required()
    def delete_feedback(feedback_id):
        try:
            current_user_id = get_jwt_identity()

            feedback = Feedback.query.filter_by(id=feedback_id, user_id=int(current_user_id)).first()

            if not feedback:
                return jsonify({'code': 404, 'message': '反馈不存在或无权限删除', 'data': None}), 404

            db.session.delete(feedback)
            db.session.commit()

            print(f"✅ 反馈删除成功, ID: {feedback_id}, 用户: {current_user_id}")

            return jsonify({'code': 200, 'message': '删除成功', 'data': {'id': feedback_id}}), 200

        except Exception as e:
            print(f"删除反馈异常: {str(e)}")
            db.session.rollback()
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 管理员接口 - 获取各部门待处理反馈（按部门分组）====================
    @app.route('/api/admin/department/pending', methods=['GET'])
    @jwt_required()
    def get_department_pending_by_dept():
        """
        获取各部门待处理反馈，按部门分组返回，已强制清洗所有字段
        """
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.get(current_user_id)

            if not current_user or current_user.user_type != 'admin':
                return jsonify({'code': 403, 'message': '权限不足', 'data': None}), 403

            pending_feedbacks = Feedback.query.filter(
                Feedback.status.in_(['pending_manual', 'processing'])
            ).order_by(Feedback.created_at.desc()).all()

            department_data = {
                'agriculture': {'count': 0, 'list': []},
                'infrastructure': {'count': 0, 'list': []},
                'environment': {'count': 0, 'list': []},
                'health': {'count': 0, 'list': []},
                'civil': {'count': 0, 'list': []},
                'general': {'count': 0, 'list': []}
            }

            category_map = {
                '农业服务': 'agriculture',
                '基建维修': 'infrastructure',
                '环境卫生': 'environment',
                '医疗卫生': 'health',
                '民政咨询': 'civil',
                '其他': 'general'
            }

            for fb in pending_feedbacks:
                dept_id = fb.category
                if not dept_id or dept_id.strip() == '':
                    dept_id = 'general'
                if dept_id in category_map:
                    dept_id = category_map[dept_id]
                if dept_id not in department_data:
                    dept_id = 'general'

                confidence = fb.ai_confidence or 0

                ai_tags = []
                if fb.ai_result:
                    if isinstance(fb.ai_result, dict):
                        ai_tags = fb.ai_result.get('aiTags', [])
                    elif isinstance(fb.ai_result, list):
                        ai_tags = fb.ai_result

                raw_voice = fb.voice_text
                clean_voice = ''
                if raw_voice is not None and str(raw_voice).strip() != '':
                    clean_voice = str(raw_voice).replace('undefined', '').replace('None', '').strip()

                clean_time = ''
                if fb.created_at is not None:
                    try:
                        clean_time = fb.created_at.isoformat()
                    except:
                        clean_time = ''

                department_data[dept_id]['count'] += 1
                department_data[dept_id]['list'].append({
                    'id': fb.id,
                    'title': fb.title,
                    'content': fb.content,
                    'confidence': confidence,
                    'status': fb.status,
                    'manual_category': fb.manual_category,
                    'voice_url': fb.voice_url or '',
                    'voice_text': clean_voice,
                    'voice_duration': fb.voice_duration or 0,
                    'ai_tags': ai_tags[:3],
                    'created_at': clean_time
                })

            print("SAMPLE:", json.dumps(department_data['general']['list'][0] if department_data['general']['list'] else {}))

            return jsonify({'code': 200, 'message': '获取成功', 'data': department_data}), 200

        except Exception as e:
            print(f"获取部门待处理异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500


    # ==================== 管理员接口 - 获取部门待处理计数（汇总统计）====================
    @app.route('/api/admin/department/pending-count', methods=['GET'])
    @jwt_required()
    def get_department_pending_count():
        """
        获取各部门待处理计数，用于首页展示
        功能：返回各部门待处理总数和待人工审核数量
        """
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.get(current_user_id)

            if not current_user or current_user.user_type != 'admin':
                return jsonify({'code': 403, 'message': '权限不足', 'data': None}), 403

            # 查询所有待处理的反馈
            pending_feedbacks = Feedback.query.filter(
                Feedback.status.in_(['pending_manual', 'processing'])
            ).all()

            total_count = len(pending_feedbacks)
            
            # 计算低置信度（需要人工复核）的数量
            ai_threshold = 0.7
            need_review_count = sum(1 for fb in pending_feedbacks if (fb.ai_confidence or 0) < ai_threshold)

            # 按部门统计
            by_department = {}
            dept_names = ['agriculture', 'infrastructure', 'environment', 'health', 'civil', 'general']
            
            for dept in dept_names:
                count = sum(1 for fb in pending_feedbacks if fb.category == dept)
                if count > 0:
                    by_department[dept] = count

            return jsonify({
                'code': 200,
                'message': '获取成功',
                'data': {
                    'total': total_count,
                    'need_review': need_review_count,
                    'by_department': by_department
                }
            }), 200

        except Exception as e:
            print(f"获取部门待处理计数异常: {str(e)}")
            return jsonify({'code': 500, 'message': str(e), 'data': None}), 500

    # ==================== 提交反馈接口（带AI分类）====================
    @app.route('/api/feedback/submit', methods=['POST'])
    @jwt_required()
    def submit_feedback():
        try:
            current_user_id = get_jwt_identity()
            data = request.get_json()
            
            print(f"📝 收到反馈提交: 用户 {current_user_id}")
            
            if not data.get('title'):
                return jsonify({'code': 400, 'message': '反馈标题不能为空', 'data': None}), 400
            
            if not data.get('content'):
                return jsonify({'code': 400, 'message': '反馈内容不能为空', 'data': None}), 400
            
            user = User.query.get(int(current_user_id))
            if not user:
                return jsonify({'code': 404, 'message': '用户不存在', 'data': None}), 404
            
            title = data.get('title', '')
            content = data.get('content', '')
            
            voice_url = data.get('voiceUrl', '')
            voice_text = data.get('voiceText', '')
            voice_duration = data.get('voiceDuration', 0)
            
            full_analysis_text = title + ' ' + content
            if voice_text:
                full_analysis_text += ' ' + voice_text
                print(f"🎤 包含语音转写内容: {voice_text}")
            
            ai_result = ai_service.classify_feedback(full_analysis_text)
            print(f"🤖 AI分类结果: {ai_result}")
            
            confidence = ai_result.get('confidence', 0)
            category = ai_result.get('category', 'general')
            need_manual = ai_result.get('need_manual', confidence < 0.7)
            
            # 如果是 general 分类（其他），强制进入人工分类
            if category == 'general':
                need_manual = True
                status = 'pending_manual'
                print(f"📋 分类为'其他'，强制进入人工分类")
            elif voice_url and confidence < 0.8:
                need_manual = True
                status = 'pending_manual'
                print(f"🎤 包含语音文件且置信度{confidence}<0.8，强制进入人工分类")
            elif not need_manual and confidence >= 0.7:
                status = 'processing'
                print(f"✅ AI分类成功 (置信度: {confidence})，直接进入处理中状态")
            else:
                status = 'pending_manual'
                print(f"⚠️ AI置信度较低 (置信度: {confidence})，需要人工审核")
            
            feedback = Feedback(
                user_id=int(current_user_id),
                title=title,
                content=content,
                category=ai_result['category'],
                status=status,
                ai_confidence=confidence,
                ai_result=ai_result,
                images=data.get('imageUrls'),
                voice_url=voice_url,
                voice_text=voice_text,
                voice_duration=voice_duration
            )
            
            db.session.add(feedback)
            db.session.commit()
            
            print(f"✅ 反馈保存成功, ID: {feedback.id}, 用户: {user.username}, 状态: {status}")
            print(f"   📁 语音文件: {voice_url if voice_url else '无'}")
            
            return_data = {
                'id': feedback.id,
                'status': feedback.status,
                'status_display': '处理中' if status == 'processing' else '待分类',
                'ai_category': ai_result['category'],
                'ai_label': ai_result.get('label', '其他'),
                'ai_confidence': confidence,
                'need_manual': need_manual,
                'has_voice': bool(voice_url),
                'created_at': feedback.created_at.isoformat() if feedback.created_at else None
            }
            
            if status == 'processing':
                dept_names = {
                    'agriculture': '农业服务部',
                    'infrastructure': '基建维修部',
                    'environment': '环境整治部',
                    'health': '医疗卫生部',
                    'civil': '民政服务部',
                    'general': '综合服务部'
                }
                return_data['department'] = dept_names.get(ai_result['category'], '综合服务部')
                return_data['department_id'] = ai_result['category']
            
            return jsonify({'code': 200, 'message': '反馈提交成功', 'data': return_data}), 200
            
        except Exception as e:
            print(f"❌ 提交反馈异常: {str(e)}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== AI分类专用接口 ====================
    @app.route('/api/ai/classify', methods=['POST'])
    @jwt_required()
    def ai_classify():
        try:
            data = request.get_json()
            content = data.get('content', '')
            
            if not content:
                return jsonify({'code': 400, 'message': '内容不能为空', 'data': None}), 400
            
            ai_result = ai_service.classify_feedback(content)
            
            return jsonify({
                'code': 200,
                'message': '分类成功',
                'data': {
                    'category': ai_result['category'],
                    'label': ai_result['label'],
                    'dept_name': ai_result.get('dept_name', '综合服务部'),
                    'confidence': ai_result['confidence'],
                    'need_manual': ai_result['need_manual'],
                    'aiTags': ai_result.get('aiTags', [])
                }
            }), 200
            
        except Exception as e:
            print(f"AI分类接口异常: {str(e)}")
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== AI对话接口 ====================
    @app.route('/api/ai/chat', methods=['POST'])
    @jwt_required()
    def ai_chat():
        try:
            data = request.get_json()
            question = data.get('question', '')
            
            if not question:
                return jsonify({'code': 400, 'message': '问题不能为空', 'data': None}), 400
            
            print(f"收到咨询：{question}")
            
            answer, error = ai_service.chat(question)
            
            if error:
                print(f"AI对话失败: {error}")
                return jsonify({'code': 503, 'message': f'AI服务暂时不可用', 'data': None}), 503
            
            return jsonify({'code': 200, 'message': 'success', 'data': {'answer': answer}}), 200
            
        except Exception as e:
            print(f"AI咨询异常: {str(e)}")
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 语音识别接口 ====================
    @app.route('/api/voice/recognize', methods=['POST'])
    @jwt_required()
    def voice_recognize():
        print("=" * 60)
        print("🎤 语音识别接口被调用")
        print("=" * 60)
        
        try:
            current_user_id = get_jwt_identity()
            print(f"1. 用户ID: {current_user_id}")
            
            if 'voice' not in request.files:
                return jsonify({'code': 400, 'message': '没有上传语音文件'}), 400
            
            file = request.files['voice']
            print(f"2. 收到文件: {file.filename}")
            
            dialect = request.form.get('dialect', 'auto')
            model_size = request.form.get('model_size', 'small')
            print(f"3. 参数: dialect={dialect}, model_size={model_size}")
            
            if file.filename == '':
                return jsonify({'code': 400, 'message': '文件名为空'}), 400
            
            allowed_formats = {'mp3', 'wav', 'm4a', 'mp4', 'webm', 'mpga', 'ogg', 'aac'}
            ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            if ext not in allowed_formats:
                return jsonify({'code': 400, 'message': f'不支持的音频格式: {ext}'}), 400
            
            upload_folder = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            unique_filename = f"voice_{uuid.uuid4().hex}.{ext}"
            temp_filepath = os.path.join(upload_folder, unique_filename)
            file.save(temp_filepath)
            
            file_size = os.path.getsize(temp_filepath)
            print(f"4. 文件已保存: {file_size} 字节")
            
            if file_size < 2000:
                os.remove(temp_filepath)
                return jsonify({'code': 400, 'message': '录音太短'}), 400
            
            if not WHISPER_AVAILABLE:
                return jsonify({'code': 500, 'message': 'Whisper 未安装'}), 500
            
            # ========== 统一使用原始 Whisper 模型 ==========
            print(f"5. 使用原始 Whisper 模型识别")
            
            whisper_model = get_whisper_model(model_size, use_finetuned=False)
            
            if not whisper_model:
                return jsonify({'code': 500, 'message': '模型加载失败'}), 500
            
            options = {
                'fp16': False,
                'task': 'transcribe'
            }
            
            # 粤语和普通话都指定为中文
            if dialect == 'cantonese':
                options['language'] = 'zh'
                print("   🗣️ 粤语识别模式")
            else:
                options['language'] = 'zh'
                print("   🗣️ 普通话识别模式")
            
            result = whisper_model.transcribe(temp_filepath, **options)
            recognize_text = result.get('text', '').strip()
            detected_language = result.get('language', 'zh')
            
            print(f"6. 原始识别结果: '{recognize_text}'")
            
            file_url = f"http://localhost:5000/uploads/{unique_filename}"
            
            if recognize_text and len(recognize_text) > 0:
                try:
                    recognize_text = zhconv.convert(recognize_text, 'zh-hans')
                except:
                    pass
                
                corrected_text, need_confirm = semantic_correction_with_llm(recognize_text)
                final_text = corrected_text if corrected_text else recognize_text
                
                print(f"7. 最终识别文本: '{final_text}'")
                
                return jsonify({
                    'code': 200,
                    'message': '识别成功',
                    'data': {
                        'text': final_text,
                        'voice_url': file_url,
                        'language': detected_language,
                        'language_display': '粤语' if dialect == 'cantonese' else '普通话',
                        'need_confirm': need_confirm
                    }
                })
            else:
                return jsonify({
                    'code': 200,
                    'message': '未能识别出有效内容',
                    'data': {'text': '', 'voice_url': file_url}
                })
                
        except Exception as e:
            print(f"语音识别异常: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'code': 500, 'message': str(e)}), 500
  

    # ==================== 图片上传接口 ====================
    @app.route('/api/upload/image', methods=['POST'])
    @jwt_required()
    def upload_image():
        try:
            upload_folder = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            if 'image0' not in request.files:
                return jsonify({'code': 400, 'message': '没有上传文件', 'data': None}), 400
            
            file = request.files['image0']
            if file.filename == '':
                return jsonify({'code': 400, 'message': '文件名为空', 'data': None}), 400
            
            if not allowed_file(file.filename, {'png', 'jpg', 'jpeg', 'gif'}):
                return jsonify({'code': 400, 'message': '不支持的文件类型', 'data': None}), 400
            
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(upload_folder, filename)
            
            file.save(filepath)
            
            file_url = f"http://localhost:5000/uploads/{filename}"
            
            return jsonify({'code': 200, 'message': '上传成功', 'data': {'url': file_url}}), 200
            
        except Exception as e:
            print(f"图片上传异常: {str(e)}")
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 语音上传接口（备用）====================
    @app.route('/api/upload/voice', methods=['POST'])
    @jwt_required()
    def upload_voice():
        try:
            upload_folder = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            if 'voice' not in request.files:
                return jsonify({'code': 400, 'message': '没有上传文件', 'data': None}), 400
            
            file = request.files['voice']
            if file.filename == '':
                return jsonify({'code': 400, 'message': '文件名为空', 'data': None}), 400
            
            if not allowed_file(file.filename, {'mp3', 'wav', 'm4a'}):
                return jsonify({'code': 400, 'message': '不支持的音频格式', 'data': None}), 400
            
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(upload_folder, filename)
            
            file.save(filepath)
            
            file_url = f"http://localhost:5000/uploads/{filename}"
            
            return jsonify({'code': 200, 'message': '上传成功', 'data': {'url': file_url}}), 200
            
        except Exception as e:
            print(f"语音上传异常: {str(e)}")
            return jsonify({'code': 500, 'message': '服务器内部错误', 'data': None}), 500

    # ==================== 静态文件访问接口 ====================
    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        upload_folder = app.config['UPLOAD_FOLDER']
        return send_from_directory(upload_folder, filename)
    
    # ==================== 公告列表接口 ====================
    @app.route('/api/common/noticeList', methods=['GET'])
    def get_notice_list():
        mock_notices = [
            { 'id': 1, 'title': '春节期间服务时间调整通知', 'date': '2026-02-01', 'content': '春节期间（2月6日-2月12日）乡镇服务中心调整为上午9:00-11:30开放' },
            { 'id': 2, 'title': '农业补贴政策解读会通知', 'date': '2026-02-03', 'content': '定于2月8日上午9点在村委会举办2026年度农业补贴政策解读会' },
            { 'id': 3, 'title': '乡村道路维修工程公告', 'date': '2026-02-05', 'content': '王家村至李家村道路将于2月10日开始维修，预计工期15天' }
        ]
        return jsonify({'code': 200, 'data': mock_notices, 'message': 'success'}), 200
    
# ==================== 启动应用 ====================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 正在启动乡镇服务反馈平台后端服务器...")
    print("="*50)
    
    app = create_app()
    
    print("\n✅ 后端服务器启动成功！")
    print("📡 访问地址: http://localhost:5000")
    print("🔗 API测试: http://localhost:5000/api/auth/test")
    print("📊 管理员统计: http://localhost:5000/api/admin/statistics")
    print("👥 待审核用户: http://localhost:5000/api/admin/users/pending")
    print("📨 消息接口: http://localhost:5000/api/messages")
    print("📤 上传接口: http://localhost:5000/api/upload/image")
    print("🤖 AI分类接口: http://localhost:5000/api/feedback/submit")
    print("🤖 AI对话接口: http://localhost:5000/api/ai/chat")
    print("📋 反馈列表接口: http://localhost:5000/api/feedback/list")
    print("📋 反馈详情接口: http://localhost:5000/api/feedback/detail/1")
    print("📋 部门待处理接口: http://localhost:5000/api/admin/department/pending")
    print("📋 部门待处理计数: http://localhost:5000/api/admin/department/pending-count")
    print("🎤 语音识别接口: http://localhost:5000/api/voice/recognize (Whisper + Qwen语义修正)")
    
    print("\n🔄 按 Ctrl+C 停止服务器")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)