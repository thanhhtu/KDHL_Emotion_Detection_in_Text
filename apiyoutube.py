import googleapiclient.discovery
import csv
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
DEV_KEY = "AIzaSyBt13wqYq9jZ8w8XjWzDPxLS64YOhqBA-M"

youtube = googleapiclient.discovery.build(API_SERVICE_NAME, API_VERSION, developerKey=DEV_KEY)


video_id = "Rgazx347CwQ"


# Mở file CSV để ghi
with open("youtube_comments.csv", mode="w", encoding="utf-8", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Author", "Comment"])  # Header


    # Lấy comment bằng API
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=200,   # tối đa 100 comment mỗi lần gọi
        textFormat="plainText"
    )


    while request:
        response = request.execute()


        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            author = comment["authorDisplayName"]
            text = comment["textDisplay"]
            writer.writerow([author, text])


        # Di chuyển sang trang tiếp theo nếu có
        if "nextPageToken" in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response["nextPageToken"],
                textFormat="plainText"
            )
        else:
            break  




print("✅ Đã lưu comment vào youtube_comments.csv")