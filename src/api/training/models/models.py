import time

from projectdavid_common.schemas.enums import StatusEnum
from sqlalchemy import JSON, BigInteger, Boolean, Column
from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()
