# Creating 8 lists corresponding to the categories
from huggingface_hub import hf_hub_download, snapshot_download, login
from OmniParser.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

login("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")

parsed_content_list = ""
context_description = (
    f"You are given a website screenshot with elements that have been bounded by boxes to increase precision. "
    f"Here is the list of the ground truth bounding boxes on this screen and their corresponding elements:\n"
    f"{parsed_content_list}\n"
)

box_prop = [['Text Box ID 0: Q Tim kiem tren Facebook', 'Text Box ID 1: Phat Tran', 'Text Box ID 2: Tran oi, an dang nghi gi the?', 'Text Box ID 3: Dudc tai tro', 'Text Box ID 4: Ban be (85 nguoi online)', "Text Box ID 5: O1 Video try'c tiep", 'Text Box ID 6:  Anh/video', 'Text Box ID 7: O Cam xuc/hoat dong', 'Text Box ID 8: Batdongsan.com.vn', 'Text Box ID 9: veoor', 'Text Box ID 10: Ky niem', 'Text Box ID 11: batdongsan.com.vn', 'Text Box ID 12: Da luu', 'Text Box ID 13: THACTAP INH', 'Text Box ID 14: Nhom', "Text Box ID 15: Chu'ong trinh Thy'c tap", 'Text Box ID 16: sinh tai nang', 'Text Box ID 17: D', 'Text Box ID 18: Video', 'Text Box ID 19: docs.google.com', 'Text Box ID 20: Nguyen Tan', 'Text Box ID 21: Pham Duc', 'Text Box ID 22: Ngo Gia Phuc', 'Text Box ID 23: guyen', 'Text Box ID 24: Tao tin', 'Text Box ID 25: Thanh', 'Text Box ID 26: Binh', 'Text Box ID 27: Marketplace', 'Text Box ID 28: Sinh nhat', 'Text Box ID 29: Binh Dan Hoc Al', 'Text Box ID 30: Bang feed', 'Text Box ID 31: Luu Trung Vu 7 phut O', 'Text Box ID 32: Hom nay la sinh nhat cua Quang Bao.', 'Text Box ID 33: Cac bac di cho e hoi con Al nao tao may anh nay vs a va neu tao thi promt sao cho chuan a!E', 'Text Box ID 34: V', 'Text Box ID 35: Xem them', 'Text Box ID 36: cam dn a', "Text Box ID 37: Ngu'di lien he.", 'Text Box ID 38: 105', 'Text Box ID 39: Loi tat cua ban', 'Text Box ID 40: Quan Nguyen', 'Text Box ID 41: Viet Referral', 'Text Box ID 42: Cao Nguyen Nguyen', 'Text Box ID 43: Viet Tech', 'Text Box ID 44: Nguyen Ly Gia Binh', 'Text Box ID 45: Binh Dan Hoc Al', "Text Box ID 46: Du'ong Quang Trung", 'Text Box ID 47: 12', 'Text Box ID 48: Forum Machine Learning co bane', 'Text Box ID 49: Nguyen Minh Khanh', 'Text Box ID 50: Vi Le', 'Text Box ID 51: Duong Hoang Thy Thy', 'Text Box ID 52: Quyen rieng tu Dieu khoan - Quang cao - Lua chon', 'Text Box ID 53: Thanh Thao', 'Text Box ID 54: quang cao [> : Cookie - Xem them - Meta  2024', 'Text Box ID 55: Muontrong', 'Text Box ID 56: Mot', 'Text Box ID 57: xihoi', 'Text Box ID 58: dau da dau', 'Icon Box ID 59: a chat or messaging application.', 'Icon Box ID 60: a grid-like layout for organizing items.', 'Icon Box ID 61: a person paddling a canoe or kayak.', 'Icon Box ID 62: a notification or alert.', 'Icon Box ID 63: a user profile or profile picture.', 'Icon Box ID 64: a logo for a technology company named Vistech.', 'Icon Box ID 65: an internship application.', 'Icon Box ID 66: a pen or writing tool.', 'Icon Box ID 67: a user profile or profile picture.', 'Icon Box ID 68: a credit or debit card.', 'Icon Box ID 69: a bookmarks or bookmarking feature.', 'Icon Box ID 70: a group of people or users.', 'Icon Box ID 71: a button to add or create a new item.', 'Icon Box ID 72: a clock showing the current time.', 'Icon Box ID 73: a shopping or retail store.', 'Icon Box ID 74: a person on a boat with a mountain in the background.', 'Icon Box ID 75: a shopping or retail store.', 'Icon Box ID 76: a user profile or account.', 'Icon Box ID 77: a person paddling a kayak on a river.', 'Icon Box ID 78: a person paddling a boat on a river.', 'Icon Box ID 79: a photo of a couple.', 'Icon Box ID 80: a user profile or profile picture.', 'Icon Box ID 81: a user profile or user account.', 'Icon Box ID 82: a user profile with a green checkmark.', 'Icon Box ID 83: a business card or contact information.', 'Icon Box ID 84: a house or home.', 'Icon Box ID 85: a video player or media player.', 'Icon Box ID 86: a motorcycle with the name "khoa le".', 'Icon Box ID 87: a "X" or "Close" button.', 'Icon Box ID 88: a Facebook logo.', 'Icon Box ID 89: a user profile or profile picture.', 'Icon Box ID 90: a gift or present.', 'Icon Box ID 91: a person walking on a sidewalk.', 'Icon Box ID 92: the word "duong" or "doung".', 'Icon Box ID 93: a person wearing a watch.', 'Icon Box ID 94: a user profile or profile picture.', 'Icon Box ID 95: a couple taking a selfie.', 'Icon Box ID 96: a loading or progress bar.', 'Icon Box ID 97: a user profile or profile picture.', 'Icon Box ID 98: a family with a child and a dog.', 'Icon Box ID 99: a vertical scroll bar.'],
['Text Box ID 0: Gmail', 'Text Box ID 1: Images', 'Text Box ID 2: 4', 'Text Box ID 3: Google', 'Text Box ID 4:  Search Google or type a URL', 'Text Box ID 5: x', 'Text Box ID 6: [m]', 'Text Box ID 7: Facebook', 'Text Box ID 8: (757) YouTube', 'Text Box ID 9: NimoTV-Top ...', 'Text Box ID 10: x.com', 'Text Box ID 11:  Instagram', 'Text Box ID 12: ChatGPT', 'Text Box ID 13: Kaggle', 'Text Box ID 14: Claude', 'Text Box ID 15: Papers With ...', 'Text Box ID 16: Add shortcut', 'Text Box ID 17:  Customize Chrome', 'Icon Box ID 18: a microphone and a camera.', 'Icon Box ID 19: a person sitting on a bench with a tree in the background.', 'Icon Box ID 20: a grid of dots.'],
['Text Box ID 0:  deeple', 'Text Box ID 1: notel', 'Text Box ID 2: icon', 'Text Box ID 3: 0RTX 4', 'Text Box ID 4: e2301.', 'Text Box ID 5: 2408.', 'Text Box ID 6: e2402.0', 'Text Box ID 7: deskt', 'Text Box ID 8: D (7 4) x', 'Text Box ID 9: EH ST UEI x', 'Text Box ID 10: loginst.ueh.edu.vn/?returnURL=http://stude', 'Text Box ID 11: UEH', 'Text Box ID 12: UNIVERSITY', 'Text Box ID 13: DANH CHO nGUOI HOC', 'Text Box ID 14: Ma so sinh vien', 'Text Box ID 15: 1211027319', 'Text Box ID 16: Matkhau', 'Text Box ID 17:  Xem huong dan', 'Text Box ID 18:  Luu dang nhgp tren thiet bj ndy.', 'Text Box ID 19: Quen mgt khau', 'Text Box ID 20: DANG NHAP', 'Text Box ID 21: G Nhan vao day de dang nhap bang Google voi Email sT UEH', 'Text Box ID 22: UEH', 'Icon Box ID 23: a smiley face emoticon.', 'Icon Box ID 24: a person sitting on a bench with a tree in the background.', 'Icon Box ID 25: a smiley face with the text "Metadata".', 'Icon Box ID 26: a horizontal scroll bar.', 'Icon Box ID 27: a smiley face emoticon.', 'Icon Box ID 28: a dropdown menu with three options.', 'Icon Box ID 29: adding or creating a new item.', 'Icon Box ID 30: the "Privacy - Terms" option.', 'Icon Box ID 31: the Global Positioning System or GPS.', 'Icon Box ID 32: a Facebook logo.', 'Icon Box ID 33: a key or unlock function.', 'Icon Box ID 34: a "X" or "Close" button.', 'Icon Box ID 35: a network or data connection.', 'Icon Box ID 36: a "X" or "Close" button.', 'Icon Box ID 37: a "X" or "No" button.', 'Icon Box ID 38: a "back" or "previous" action.', 'Icon Box ID 39: the "DataBackup" function.', 'Icon Box ID 40: a home button with the text "2F Home".', 'Icon Box ID 41: a return to the previous screen.', 'Icon Box ID 42: a "X" or "Close" button.', 'Icon Box ID 43: a loading indicator or progress bar.', 'Icon Box ID 44: a "back" or "previous" action.'],
['Text Box ID 0: O deeple', 'Text Box ID 1:  k noteb', 'Text Box ID 2:  k icon_', 'Text Box ID 3: 0 RTX 4', 'Text Box ID 4: 0GPT-4X', 'Text Box ID 5: Elo.dr', 'Text Box ID 6: Faceb', 'Text Box ID 7: @Huggi', 'Text Box ID 8: @ openbX', 'Text Box ID 9:  meta', 'Text Box ID 10: 2 2301.1', 'Text Box ID 11: P 2408.0', 'Text Box ID 12: e 2402.0', 'Text Box ID 13: desktc', 'Text Box ID 14: Bisko', 'Text Box ID 15: e6 tiktok.com/@bisko.freedom?fbc|id=IwZXh0bgNhZW0CMTAAAR0PtZ5SmC5VgH_tgi4EavkskB3J1RIae3M9weWCIfR_N9dyb7KPqdERak_aem_AQwERzH-gr727jaLhsGS0zbK2chmK7Whps7RqDeVTS4bmJPC-_qi8JvFiV3-bjq|5poAqzIILmXo-vks..', 'Text Box ID 16: dTikTok', 'Text Box ID 17: Search', 'Text Box ID 18: Q', 'Text Box ID 19: Log in', 'Text Box ID 20: For You', 'Text Box ID 21: bisko.freedomBisko', 'Text Box ID 22: O Explore', 'Text Box ID 23: Follow', 'Text Box ID 24: Message', 'Text Box ID 25: 3oFollowing', 'Text Box ID 26: 21 Following188.9K Followers', 'Text Box ID 27: 2.2M LikeS', 'Text Box ID 28: CLIVE', 'Text Box ID 29: vN I eat eggs and go on adventures vN', 'Text Box ID 30:  www.patreon.com/bisko', 'Text Box ID 31: oProfile', 'Text Box ID 32:  III Vides', 'Text Box ID 33: t1 Reposts', 'Text Box ID 34: (3 Liked', 'Text Box ID 35:  LatestPopularOldest', 'Text Box ID 36: Log in to follow creators,', 'Text Box ID 37: like videos, and view', 'Text Box ID 38: comments.', 'Text Box ID 39: Playlists', 'Text Box ID 40: Log in', 'Text Box ID 41: Ben Tre N', 'Text Box ID 42: Tuyen Quang vN', 'Text Box ID 43: VUung Tau vN', 'Text Box ID 44: 41 posts', 'Text Box ID 45: 3 p0sts', 'Text Box ID 46: 26 posts', 'Text Box ID 47: Create TIkTok efects,', 'Text Box ID 48:  get a reward', 'Text Box ID 49: Videos', 'Text Box ID 50: LIVE LIVESTREAM ', 'Text Box ID 51: Company', 'Text Box ID 52: UNCLE IS HEALTHY ', "Text Box ID 53: IT's COOD FOR THE BIRD ", 'Text Box ID 54: PATRIOTIC ENGLISH CLAS I', 'Text Box ID 55: EG TEA ', 'Text Box ID 56: GRANDMA RIVER ', 'Text Box ID 57: Program', 'Text Box ID 58: Terms & Policies', 'Text Box ID 59: @ 2024 TikTok', 'Text Box ID 60: VIETNAM', 'Text Box ID 61: Filming Tiktok', 'Text Box ID 62: Healthy!', 'Text Box ID 63: Handsome #', "Text Box ID 64: What's Grandma's Name?", 'Text Box ID 65: Dang QuayTiktok', 'Text Box ID 66: Khoe Manh!', 'Text Box ID 67: Dep Trai #1 8Day', 'Text Box ID 68: 1Ten Ba Gi?', 'Text Box ID 69: > 366.3K', 'Text Box ID 70: > 166.5K', 'Text Box ID 71: > 4.1M', 'Text Box ID 72: D 185K', 'Text Box ID 73: > 155.2K', 'Text Box ID 74: D 284.7K', 'Text Box ID 75: FREE WIFE', 'Text Box ID 76: COMF HOMEDUC HUY', 'Text Box ID 77: FREE SHAVE ', 'Text Box ID 78: WAIKING IO BA IRI', 'Text Box ID 79: ONIY 1 BOITIF', 'Text Box ID 80: HOMESTAY REVIEW', 'Icon Box ID 81: a text input field.', 'Icon Box ID 82: an arrow pointing to the next item or action.', 'Icon Box ID 83: a download or save function.', 'Icon Box ID 84: a YouTube video playing in the background.', 'Icon Box ID 85: a forward or next action.', 'Icon Box ID 86: a music note.', 'Icon Box ID 87: a person wearing glasses.', 'Icon Box ID 88: a "refresh" or "reload" function.', 'Icon Box ID 89: a horizontal scroll bar.', 'Icon Box ID 90: a network or data connection.', 'Icon Box ID 91: adding or creating a new item.', 'Icon Box ID 92: a user profile or profile picture.', 'Icon Box ID 93: a document or file.', 'Icon Box ID 94: a user profile or profile picture.', 'Icon Box ID 95: a person holding a bottle of beer, with the text "Cheers, it\'s a beer."', 'Icon Box ID 96: an airplane flying in the sky.', 'Icon Box ID 97: a vertical scroll bar.', 'Icon Box ID 98: a loading or buffering indicator.', 'Icon Box ID 99: a horizontal line or border.', 'Icon Box ID 100: a person sitting on a bench under a tree.', 'Icon Box ID 101: a user profile or profile picture.', 'Icon Box ID 102: a toggle switch in the "on" position.', 'Icon Box ID 103: a "back" or "previous" action.', 'Icon Box ID 104: a person wearing sunglasses.', 'Icon Box ID 105: a person standing in front of a statue.', 'Icon Box ID 106: a loading or buffering indicator.', 'Icon Box ID 107: a toggle switch in the "on" position.'],
['Text Box ID 0: O deepl', 'Text Box ID 1: noteb', 'Text Box ID 2: 0 RTX 4', 'Text Box ID 3: $GPT-4', 'Text Box ID 4: Elo.dr', 'Text Box ID 5: meta', 'Text Box ID 6: 2301.', 'Text Box ID 7: 22408.', 'Text Box ID 8: 2402.(', 'Text Box ID 9: deskt', 'Text Box ID 10: (74', 'Text Box ID 11:  student. ueh.edu.vn/Home', 'Text Box ID 12: UEH', 'Text Box ID 13: UNIVERSITY', 'Text Box ID 14: Trang chu', 'Text Box ID 15:  Nganh - Chuong trinh dao tao', 'Text Box ID 16: Tra ciru van bang', 'Text Box ID 17: Tra ciru hoc phan', 'Text Box ID 18: Ho tro', 'Text Box ID 19: 31211027319 | Tran Nguyen Ngoc Phat ', "Text Box ID 20:  CHU'C NANG", 'Text Box ID 21: O Thong bao', 'Text Box ID 22: Tieu de', 'Text Box ID 23: Ngwoi gi', 'Text Box ID 24: Thoi gian gi', 'Text Box ID 25: > Trang ca nhan', 'Text Box ID 26: Thong bao nop ching chi tieng Anh quoc te xet tot nghiep Khoa 42, 43.', 'Text Box ID 27: Ngo Thj Lan', 'Text Box ID 28: 14/10/2024', 'Text Box ID 29: Thong tin ca nhan', 'Text Box ID 30:  44, 45, 46, 47 - Dai hoc chinh quy, dgt 4 nam 2024', 'Text Box ID 31: ...T hong bao (24)', 'Text Box ID 32: [SHcD] Thong bao dot hoc Sinh hoat cong dan nam 2024', 'Text Box ID 33: Nguyen Cong Nam', 'Text Box ID 34: 31/08/2024', 'Text Box ID 35: > Tra ciru thong tin', 'Text Box ID 36: [RLSv] Thong bao trien khai danh gia ket qua ren luyen sinh vien Hoc ky', 'Text Box ID 37: Nguyen Cong Nam', 'Text Box ID 38: 13/08/2024', 'Text Box ID 39: Cuoi nam 2024', 'Text Box ID 40: .. Churong trinh dao tao', 'Text Box ID 41: Thong bao xet chuyen diem, mien hoc phan tieng Anh va chuan dau ra', 'Text Box ID 42: Ngo Thj Lan', 'Text Box ID 43:  09/08/2024', 'Text Box ID 44: Lich hoc', 'Text Box ID 45:  trinh do tieng Anh doi voi sinh vien Khoa 47, Khoa 48, Khoa 49 - DHCQ,', 'Text Box ID 46: Dot thang 9 nam 2024', 'Text Box ID 47: m Lich tihi', 'Text Box ID 48: Thong bao dang ky tham gia hoc tap trao doi tai Trwong Dai hoc Kinh te -', 'Text Box ID 49: Cu Duc Tai', 'Text Box ID 50: 22/07/2024', 'Text Box ID 51:  uyet d nh sinh vien', 'Text Box ID 52: Dai hoc Da Nang (DUE), Hgc ky 1 nam hoc 2024 - 2025', 'Text Box ID 53:  Chuyen can', "Text Box ID 54: Thong bao v/v dang ky tham gia hoc tap trao doi tai Trwo'ng Dai hoc Kinh", 'Text Box ID 55: Cu Dwc Tai', 'Text Box ID 56: 28/06/2024', 'Text Box ID 57: te Quoc dan (NEU) trong Hgc ky Thu 2024', 'Text Box ID 58:  Ket qua ren luyen', 'Text Box ID 59: Tiep nhan dang ky hoc bong trao doi sinh vien quoc te Dot 2 Hoc ky dau', 'Text Box ID 60: Trwong Nhat Uyen', 'Text Box ID 61: 17/06/2024', 'Text Box ID 62: - Ket qua hoc tap', 'Text Box ID 63: 2024', 'Text Box ID 64: .. Tai chinh sinh vien', 'Text Box ID 65: Tiep nhan dang ky hoc bong trao doi sinh vien quoc te Dot 1 Hoc ky dau', "Text Box ID 66: Trwo'ng Nhat Uyen", 'Text Box ID 67:  06/06/2024', 'Text Box ID 68: . Ch tiet noa don', 'Text Box ID 69: 2024', 'Text Box ID 70: . +em ket qua dang ky hoc phan', 'Text Box ID 71: THONG BAO DANG KY THAM GIA HOC TAP TRAO DOI TAI TRUONG DAI', 'Text Box ID 72: Cu Duc Tai', 'Text Box ID 73: 18/04/2024', 'Text Box ID 74: HOC KINH TE QUOC DAN (NEU) TRONG HOC KY HE 2024 (HOC KY GIUA', 'Text Box ID 75: . Hoc bong, Chinh sach, Mien', 'Text Box ID 76: 2024)', 'Text Box ID 77: giam, Tro cap', 'Text Box ID 78: Thong bao ve hoc bong Khuyen khich hoc tap HKD 2024 danh cho sinh', "Text Box ID 79: Trwo'ng Nhat Uyen", 'Text Box ID 80: 17/04/2024', 'Text Box ID 81:  Hoc phan turong duong', 'Text Box ID 82: vien DHCQ khoa 47, 48, 49', 'Text Box ID 83:  > Churc nang tryc tuyen', 'Text Box ID 84: THONG BAO XET CHUYEN DIEM HOC PHAN TIENG ANH VA CHUAN', 'Text Box ID 85: Ngo Thj Lan', 'Text Box ID 86: 08/03/2024', 'Text Box ID 87:  TIENG Anh DAu RA doI vOI SINh vIen khOA 47, KHOA 48 - dHCQ -', 'Text Box ID 88:  Dang ky trurong doi tac', 'Text Box ID 89: DOT THANG 3 NAM 2024', 'Text Box ID 90: . et qua dang ky vang thi', 'Text Box ID 91: [RLsv] Thong bao trien khai danh gia ket qua ren luyen sinh vien Hoc ky', 'Text Box ID 92: Nguyen Cong Nam', 'Text Box ID 93: 01/03/2024', 'Text Box ID 94: Dau nam 2024', 'Text Box ID 95: - Dang ky tham dy Ie tot nghiep', 'Text Box ID 96: Danh gia ket qua ren luyen Hoc ky cuoi 2023', 'Text Box ID 97: Nguyen Cong Nam', 'Text Box ID 98: 03/10/2023', 'Text Box ID 99: - Dang ky chuyen nganh', 'Text Box ID 100: Diem Sinh hoat cong dan nam 2023', 'Text Box ID 101:  Nguyen Cong Nam', 'Text Box ID 102: 03/10/2023', 'Icon Box ID 103: a keyboard shortcut for the "K" key.', 'Icon Box ID 104: a "refresh" or "reload" function.', 'Icon Box ID 105: a smiley face emoticon.', 'Icon Box ID 106: a smiley face emoticon.', 'Icon Box ID 107: a star icon.', 'Icon Box ID 108: a blank or unselected area.', 'Icon Box ID 109: a document or text file.', 'Icon Box ID 110: an arrow pointing to the left.', 'Icon Box ID 111: a person wearing glasses.', 'Icon Box ID 112: a button to expand or show more content.', 'Icon Box ID 113: a "X" or "Close" button.', 'Icon Box ID 114: a text input field.', 'Icon Box ID 115: a "X" or "Close" button.', 'Icon Box ID 116: an airplane flying in the sky.', 'Icon Box ID 117: a dropdown menu with three options.', 'Icon Box ID 118: a music note.', 'Icon Box ID 119: a puzzle piece with a missing piece.', 'Icon Box ID 120: a YouTube play button.', 'Icon Box ID 121: a Facebook or social media button.', 'Icon Box ID 122: a person sitting on a bench in a park.', 'Icon Box ID 123: adding or creating a new item.']]

prompts = [
[ #general description
    "What kind of website is this? Can you describe the main purpose or theme of this page?",
    "What do you see in the top navigation bar of this page?",
    "Can you describe the main sections of this page? What are their functions?",
    "What advertisements or promotions are visible on this page?",
    "What is the color scheme or overall design style of the website?"
],

[ #elements recognition
    "What elements are clickable on this page, and what do they do?",
    "Can you find any text input fields on this page? What are they labeled as?",
    "Are there any dropdown menus on this page? What options do they provide?",
    "Can you identify any buttons? What are their labels or icons?",
    "Are there any images on this page? Can you describe what they depict?"
],

[ #content specifics
    "What actions can a user perform on this page?",
    "Can you describe the search functionality on this page, if any?",
    "Are there any links to other sections of the website? Where do they lead?",
    "Is there a call-to-action on this page (e.g., 'Sign Up,' 'Learn More')? Where is it located?",
    "Are there any icons on this page? What do they represent?"
],

[ #user interaction
    "Can you describe the text in the main content area of this page?",
    "Can you find any user-generated content, such as posts or comments? What do they say?",
    "Are there any videos or interactive elements on this page? Where are they located?",
    "Is there any content that seems personalized or targeted for the user?"
],

[ #ads recognition
    "How can a user interact with this page? What actions can they take?",
    "Can you describe how a user might navigate to the next section of the page?",
    "Are there any forms on this page? What fields does the form include?",
    "If a user wanted to log in or sign up, what steps would they need to take on this page?",
    "Can you describe how a user might leave feedback or contact support on this page?"
],

[ #accessibility layout
    "Are there any advertisements on this page? What do they promote?",
    "Can you describe any banners or pop-ups visible on this page?",
    "Are there any discount offers or sales mentioned? Where are they displayed?",
    "Do you see any sponsored content? How is it marked or separated from the main content?",
    "Can you identify any brand logos or promotional materials on this page?"
],

[ #website structure
    "How is the page structured? Can you describe the layout?",
    "Are there any accessibility features visible, such as text enlargement or contrast adjustment tools?",
    "Can you identify the footer of this page? What information does it include?",
    "Is the page responsive? How does the content appear to adjust for different screen sizes?",
    "Are there any loading indicators or progress bars? What are they for?"
],
]

# test.py
#inferencing the model.
#To test:
# 1: Recognizing different parts in the image: Normal conversation (what the website is about, what can it do, what does the ads show, how many friends online,...)
# 2: Description (Short summarization of the current state of the website.)
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os
import json

class StreamingWriter:
    def __init__(self, filename):
        self.filename = filename
        self.is_first = True
        
        # Initialize the JSON file with an opening bracket
        with open(self.filename, 'w') as f:
            f.write('\n')
    
    def write_entry(self, entry):
        with open(self.filename, 'a') as f:
            if not self.is_first:
                f.write(',\n')
            f.write(entry)
            self.is_first = False
    
    def close(self):
        with open(self.filename, 'a') as f:
            f.write('\n')

writer = StreamingWriter("minicpm-testing-rawimg.txt")


model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16).to('cuda')
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

img_dir = "draft-data"
i = 0
img_files = []
for img_file in os.listdir(img_dir):
    if img_file.startswith("test-omni"):
        img_files.append(img_file)

for image_file in img_files:
    # parsed_content_list = box_prop[i]
    i += 1
    for prompt_block in prompts:
        for question in prompt_block:
            # question = context_description + question
            image_dir = os.path.join(img_dir, image_file)   
            image = Image.open(image_dir).convert('RGB')
            msgs = [{'role': 'user', 'content': [image, question]}]
            res = model.chat(
                image=None,
                msgs=msgs,
                sampling = True,
                tokenizer=tokenizer
            )
            response = f"{image_file}'\n'Generated Response: {res}"
            writer.write_entry(question)
            writer.write_entry(response)
            print(f"Question: {question}")
            print(response)
    print(f"DONE PLAYING WITH IMAGE {image_file}")
writer.close()