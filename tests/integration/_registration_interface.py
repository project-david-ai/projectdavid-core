import os

from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()
client = Entity(api_key=os.getenv("ADMIN_API_KEY"))


sacrificial_user = client.users.create_user(
    full_name="Kevin Flynn",
    email="sacrifice5@encom.com",
    is_admin=False,
)
print(sacrificial_user)


create_api_key = client.keys.create_key_for_user(
    target_user_id=sacrificial_user.id, key_name="The Grid"
)

print(create_api_key.plain_key)
