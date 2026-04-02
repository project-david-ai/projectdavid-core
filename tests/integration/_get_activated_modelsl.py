import os

from dotenv import load_dotenv
from projectdavid import Entity

# Load environment variables
load_dotenv(".tests.env")

# Initialize the ProjectDavid SDK
client = Entity(
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

# -----------------------------------------
# Lists fine tuned models , user scoped
# ------------------------------------------
models = client.models.list()
for m in models:
    print(m)

"""
Expected output:
('data', [FineTunedModelRead(id='ftm_zl0qt4QIZ8kywfl8Kez39N', user_id='user_YBAzYmHy8yvUZel8Q3fnl4', training_job_id='job_rLJld452KNqFVEMk3CeC70', name='FT: unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', description=None, base_model='unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', hf_repo=None, storage_path='models/ftm_zl0qt4QIZ8kywfl8Kez39N', is_active=False, vllm_model_id=None, status='active', created_at=1774976137, updated_at=1774976137, deleted_at=None), FineTunedModelRead(id='ftm_G05BERHAEvSRr2KTyUqWIJ', user_id='user_YBAzYmHy8yvUZel8Q3fnl4', training_job_id='job_lV7DVF0uxlZwhyO1gOU7nE', name='FT: unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', description=None, base_model='unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', hf_repo=None, storage_path='models/ftm_G05BERHAEvSRr2KTyUqWIJ', is_active=False, vllm_model_id=None, status='active', created_at=1774975840, updated_at=1775076200, deleted_at=None), FineTunedModelRead(id='ftm

('data', [FineTunedModelRead(id='ftm_zl0qt4QIZ8kywfl8Kez39N', user_id='user_YBAzYmHy8yvUZel8Q3fnl4', training_job_id='job_rLJld452KNqFVEMk3CeC70', name='FT: unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', description=None, base_model='unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', hf_repo=None, storage_path='models/ftm_zl0qt4QIZ8kywfl8Kez39N', is_active=False, vllm_model_id=None, status='active', created_at=1774976137, updated_at=1774976137, deleted_at=None), FineTunedModelRead(id='ftm_G05BERHAEvSRr2KTyUqWIJ', user_id='user_YBAzYmHy8yvUZel8Q3fnl4', training_job_id='job_lV7DVF0uxlZwhyO1gOU7nE', name='FT: unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', description=None, base_model='unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', hf_repo=None, storage_path='models/ftm_G05BERHAEvSRr2KTyUqWIJ', is_active=False, vllm_model_id=None, status='active', created_at=1774975840, updated_at=1775076200, deleted_at=None), FineTunedModelRead(id='ftm_6jDonYJbYQYLwTTfXKirma', user_id='user_YBAzYmHy8yvUZel8Q3fnl4', training_job_id='job_S7pyp24zDkbZ6JWx2aYUNt', name='FT: unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', description=None, base_model='unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', hf_repo=None, storage_path='models/ftm_6jDonYJbYQYLwTTfXKirma', is_active=False, vllm_model_id=None, status='active', created_at=1774914847, updated_at=1774914847, deleted_at=None), FineTunedModelRead(id='ftm_spRQbyQtPWyR0b8o0biBEc', user_id='user_YBAzYmHy8yvUZel8Q3fnl4', training_job_id='job_Qa12OZAHgKAuLPq43kBtnN', name='FT: unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', description=None, base_model='unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', hf_repo=None, storage_path='models/ftm_spRQbyQtPWyR0b8o0biBEc', is_active=False, vllm_model_id=None, status='active', created_at=1774914353, updated_at=1774914353, deleted_at=None), FineTunedModelRead(id='ftm_35jw4qaQNbYb2dqAfrB7N4', user_id='user_YBAzYmHy8yvUZel8Q3fnl4', training_job_id='job_0JM2GjYu47lKKAd7LJYURV', name='FT: unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', description=None, base_model='unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', hf_repo=None, storage_path='models/ftm_35jw4qaQNbYb2dqAfrB7N4', is_active=False, vllm_model_id=None, status='active', created_at=1774914232, updated_at=1774914232, deleted_at=None), FineTunedModelRead(id='ftm_7xz9mIahilV8fanocaMC48', user_id='user_YBAzYmHy8yvUZel8Q3fnl4', training_job_id='job_4KVRO2DfxBOhermYwR7PM6', name='FT: unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', description=None, base_model='unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit', hf_repo=None, storage_path='models/ftm_7xz9mIahilV8fanocaMC48', is_active=False, vllm_model_id=None, status='active', created_at=1774913520, updated_at=1774913520, deleted_at=None)])
('total', 6)

"""
