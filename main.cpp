#include <bits/stdc++.h>
#include <deque>

using namespace std;

#define MAX_DISK_NUM (10 + 1)          // 硬盘编号 1~N（N最大为10）
#define MAX_DISK_SIZE (16384 + 1)      // 每个硬盘存储单元编号 1~V（V最大为16384）
#define MAX_REQUEST_NUM (30000000 + 1) // 请求数量上限（数组下标从1开始）
#define MAX_OBJECT_NUM (100000 + 1)    // 对象数量上限（数组下标从1开始）
#define REP_NUM 3                      // 每个对象存储3个副本
#define FRE_PER_SLICING 1800           // 每个时间片段长度
#define EXTRA_TIME 105                 // 附加时间片数
#define FIRST_READ 64                  // 第一次读取操作消耗令牌
#define MAX_TAG_NUM (16 + 1)
#define BLOCK_GROUP_SIZE 510
#define TAG_THRESHOLD -0.1              // 控制判断 是否为 tag 的请求阈值，0表示就是当前请求超过 平均数，越高，表示超过平均数越多
#define LOW_SCORE_THRESHOLD 0.21        // 在处理热点请求的时候跳过低分请求 0.16
#define HOT_SAMPLE 68                   // 采样 disk 的前 HOT_SAMPLE 个 tag 请求，来判断是否是热点请求
#define SHORT_DISTANCE 100                   // 采样 disk 的前 HOT_SAMPLE 个 tag 请求，来判断是否是热点请求


struct ControlModule;
struct Disk;
struct Tag;
struct Object;
struct Request;
struct StorageUnit;

struct TupleHash {
    std::size_t operator()(const std::tuple<int, int, int>& key) const {
        int a, b, c;
        std::tie(a, b, c) = key;
        std::size_t h1 = std::hash<int>()(a);
        std::size_t h2 = std::hash<int>()(b);
        std::size_t h3 = std::hash<int>()(c);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct Block_Group
{
    int selected_type = 0;
    int start_pos; // 包含开始位置
    int end_pos;   // 包含结束位置
    int id;
    int use_size = 0;
    int constant = 0;
    int v;
    Block_Group(int id, int v) : id(id), v(v)
    {
        start_pos = (id - 1) * BLOCK_GROUP_SIZE + 1;
        end_pos = min(id * BLOCK_GROUP_SIZE, v);
    }
};

struct Object
{
    int obj_id;
    int replica[REP_NUM + 1];
    int *unit[REP_NUM + 1]; // 每个副本中对象块的存储单元索引（1-based）
    int size;
    int tag;
    int last_request_point; // 挂接未完成请求链头
    bool is_delete;
    bool *block_read_status; // 每个对象块是否已被读取（1-based）

    Object() : obj_id(0), size(0), tag(0), last_request_point(0), is_delete(false), block_read_status(nullptr)
    {
        for (int i = 1; i <= REP_NUM; i++)
        {
            unit[i] = new int[MAX_DISK_SIZE];
        }
    }
    ~Object()
    {
        for (int i = 1; i <= REP_NUM; i++)
        {
            if (unit[i])
            {
                delete[] unit[i];
                unit[i] = nullptr;
            }
        }
        if (block_read_status)
        {
            delete[] block_read_status;
            block_read_status = nullptr;
        }
    }
};

struct StorageUnit
{
    int object_id; // 0表示空闲，否则为对象ID
    int block_id;  // 对象块编号（1-based）
    StorageUnit() : object_id(0), block_id(0) {}
};

struct Disk
{
    int id;
    StorageUnit storage[MAX_DISK_SIZE];
    int head_position;
    int last_token_cost;
    int used_tokens;
    int disk_size;
    bool used_read;
    vector<int> tag_hot[MAX_TAG_NUM];
    int hot_tags[MAX_TAG_NUM];

    //========================新增block_group逻辑
    vector<Block_Group> block_groups;
    int block_group_num;

    void initBlockGroup(int num, int v)
    {
        block_group_num = num;
        block_groups.push_back(Block_Group(0, 0));
        for (int i = 1; i <= block_group_num; i++)
        {
            block_groups.push_back(Block_Group(i, v));
        }
    }

    int update_hot_tags(ControlModule* cm);

    void update_hot_tags2(ControlModule* cm);

    void update_hot_tags3(ControlModule* cm);

    Disk() : id(0), head_position(1), last_token_cost(0),
             used_tokens(0), disk_size(0)
    {
        memset(storage, 0, sizeof(storage));
    }

};

struct Request
{
    int req_id;
    int object_id;
    int prev_id;
    int arrival_time; // 请求到达时间
    bool is_done;
    float value; // 动态计算的优先级
    bool *block_read_status;

    Request() : req_id(0), object_id(0), prev_id(0), arrival_time(0), is_done(false), value(0.0), block_read_status(nullptr) {}

    ~Request()
    {
        if (block_read_status)
        {
            delete[] block_read_status;
            block_read_status = nullptr;
        }
    }
};

struct Tag
{
    int tag_id;
    int *delete_obj;
    int *write_obj;
    int *read_obj;
    double var;
    double std;
    int load;
    int peak_time;
    double score;

    Tag() : tag_id(0), delete_obj(nullptr), write_obj(nullptr), read_obj(nullptr), peak_time(0), var(0.0), std(0.0), load(0), score(0.0) {}
};

struct ControlModule
{
    Disk disks[MAX_DISK_NUM];          // 硬盘数组（下标1~N有效）
    Tag tags[MAX_TAG_NUM];             // Tag 数组（1 - 16 有效）
    Object objects[MAX_OBJECT_NUM];    // 对象数组（下标1~MAX_OBJECT_NUM-1有效）
    Request requests[MAX_REQUEST_NUM]; // 请求数组（下标1~MAX_REQUEST_NUM-1有效）

    int current_time;       // 当前时间片编号
    int T;                  // 时间片数量
    int M;                  // 标签数量
    int N;                  // 硬盘数量
    int V;                  // 每块硬盘存储单元数量
    int G;                  // 每个磁头每个时间片最大令牌数
    int req_num;
    int max_distance;

    std::unordered_map<int, int> nextTokenLookup;
    std::unordered_map<std::tuple<int, int, int>, bool, TupleHash> tokenStrategyLookup;

    ControlModule() : current_time(0), N(0), V(0), G(0), req_num(0) {}

    void updateRequestTimeouts(int new_request_id);
    void InitTokenLookupTables();
    void AssignPrimaryDisks();
    void AssignPrimaryDisks2();
};

void debug_function(ControlModule* cm, int current_time, float n1, double n2, double n3) {
    if (cm->current_time != current_time && current_time > 0) {
        return;
    }
    const std::string& filename = "/home/wangke/code_craft/compile_errors.txt";

    std::ofstream file(filename, std::ios::app); // 以追加模式打开文件
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    std::ostringstream oss;
    oss << "time: " << cm->current_time << "  参数1 : " << n1 << "  参数2 :" << n2  << "  参数3 :" << n3 << std::endl;

    file << oss.str();
    file.close();
}

void ControlModule::InitTokenLookupTables(){
    std::vector<int> candidatePrevTokens = {G, 64, 52, 42, 34, 28, 23, 19, 16, 1};

    // 初始化 Token Table
    for (int token : candidatePrevTokens) {
        int nextToken;
        if (token == 1 || token == G) {
            nextToken = 64;
        } else {
            nextToken = static_cast<int>(std::ceil(token * 0.8));
            nextToken = std::max(16, nextToken);
        }
        nextTokenLookup[token] = nextToken;
    }

    // 初始化策略查找表
    // discreteToken = shortDistance + (对象块大小 × nextTokenLookup[1])            短距离 pass
    // continueToken = (shortDistance + 对象块大小) × nextTokenLookup[prevToken]    短距离 空read
    max_distance = G / 16;
    for (int shortDistance = 1; shortDistance <= SHORT_DISTANCE; ++shortDistance) {
        for (int prevToken : candidatePrevTokens) {
            for (int blockSize = 1; blockSize <= 5; ++blockSize) {
                int continueToken = (shortDistance + blockSize) * nextTokenLookup[prevToken];
                int discreteToken = shortDistance + (blockSize * nextTokenLookup[1]);
                bool result = (discreteToken + 94 > continueToken);
                // int discreteToken = shortDistance + (blockSize * nextTokenLookup[1]);
                // bool result = (discreteToken + 94 > continueToken);
                tokenStrategyLookup[std::make_tuple(shortDistance, prevToken, blockSize)] = result;
            }
        }
    }

     std::ofstream outfile("/home/wangke/code_craft/compile_errors.txt");
     // 输出 tokenStrategyLookup 表（对象块大小 1~5 的表格）
    for (int blockSize = 1; blockSize <= 5; ++blockSize) {
        outfile << "对象块大小 = " << blockSize << "\n";
        // 表头：上一个令牌消耗
        outfile << "短距离 \\ 上一个令牌\t";
        for (int token : candidatePrevTokens) {
            outfile << token << "\t";
        }
        outfile << "\n";
        
        // 短距离：1～16
        for (int shortDistance = 1; shortDistance <= SHORT_DISTANCE; ++shortDistance) {
            outfile << shortDistance << "\t\t\t";
            for (int token : candidatePrevTokens) {
                bool val = tokenStrategyLookup[std::make_tuple(shortDistance, token, blockSize)];
                outfile << (val ? "T" : "F") << "\t";
            }
            outfile << "\n";
        }
        outfile << "\n";
    }
    
    // -----------------------
    // 输出 nextTokenLookup 表
    outfile << "上一个令牌 -> 下一个令牌 消耗 对应表:\n";
    outfile << "上一个令牌\t下一个令牌\n";
    for (int token : candidatePrevTokens) {
        outfile << token << "\t\t" << nextTokenLookup[token] << "\n";
    }
    
    outfile.close();

    int slices = (T - 1) / FRE_PER_SLICING + 1;
    for (int tag = 1; tag <= M; tag++) {
        int avg_load = 0, load = 0, max_load = 0;
        double sum = 0.0, sum_sq = 0.0;
        for (int j = 1; j <= slices; j++) {
            int val = tags[tag].read_obj[j];
            sum += val;
            sum_sq += (long long)val * val;
            avg_load += (tags[tag].write_obj[j] - tags[tag].delete_obj[j]);
            load += avg_load;
            if(avg_load > max_load) max_load = avg_load;
            if(val > tags[tag].peak_time) tags[tag].peak_time = j;
        }
        tags[tag].var = sum / T;
        double mean = sum / slices ;
        tags[tag].std = sqrt((sum_sq / slices) - (mean * mean)) / FRE_PER_SLICING;
        tags[tag].load = load / slices;
    }
    int n = 0;
    for (int tag = 1; tag <= M; tag++){
        n += tags[tag].load;
    }
    // debug_function(this, 0, n, n*3, 1111);
}

void ControlModule::AssignPrimaryDisks(){
    vector<pair<int, int>> tag_k; // 存储标签ID和需要的块组数
    
    for (int tag = 1; tag <= M; ++tag) {
        int D = (tags[tag].load + BLOCK_GROUP_SIZE - 1) / BLOCK_GROUP_SIZE * BLOCK_GROUP_SIZE; // 向上取整到100的倍数
        int k = (D + BLOCK_GROUP_SIZE - 1) / BLOCK_GROUP_SIZE; // 计算需要的块组数
        tag_k.emplace_back(tag, k);
    }

    sort(tag_k.begin(), tag_k.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // 初始化每个磁盘的空闲块组数
    int block_group_num = (V + BLOCK_GROUP_SIZE - 1) / BLOCK_GROUP_SIZE;
    int free_block_groups[MAX_DISK_NUM] = {0};
    for (int d = 1; d <= N; ++d) {
        free_block_groups[d] = block_group_num;
    }

    // 遍历每个标签进行分配
    for (const auto& elem : tag_k) {
        int tag = elem.first;
        int k = elem.second;

        vector<int> candidate_disks;
        for (int d = 1; d <= N; ++d) {
            if (free_block_groups[d] >= k) {
                candidate_disks.push_back(d);
            }
        }

        if (candidate_disks.size() < 3) continue;

        // 按剩余块组数降序排序（优先选空闲多的磁盘）
        sort(candidate_disks.begin(), candidate_disks.end(), [&](int a, int b) {
            return free_block_groups[a] > free_block_groups[b];
        });

        // 选择前三个磁盘
        int d1 = candidate_disks[0];
        // int d2 = candidate_disks[1];
        // int d3 = candidate_disks[2];

        // 更新空闲块组数
        free_block_groups[d1] -= k;

        // 标记选中磁盘的块组为当前标签
        auto mark_blocks = [&](int disk_id) {
            Disk& disk = disks[disk_id];
            int marked = 0;
            for (int bg = 1; bg <= disk.block_group_num && marked < k; ++bg) {
                if (disk.block_groups[bg].selected_type == 0) {
                    disk.block_groups[bg].selected_type = tag;
                    disk.block_groups[bg].constant = 1;
                    marked++;
                }
            }
        };
        mark_blocks(d1);
        // mark_blocks(d2);
        // mark_blocks(d3);
    }
}

struct DiskInfo {
    int free_groups;          // 剩余块组数
    vector<int> peak_times;   // 已分配标签的峰值时间
};

void ControlModule::AssignPrimaryDisks2(){
    vector<tuple<int, int, int>> tag_info; // 存储标签ID、需求块组数、峰值时间
    const int PEAK_THRESHOLD = 20; // 峰值时间冲突阈值（时间片）

    // 步骤1：收集标签信息（含峰值时间）
    for (int tag = 1; tag <= M; ++tag) {
        int D = (tags[tag].load + BLOCK_GROUP_SIZE - 1)/BLOCK_GROUP_SIZE * BLOCK_GROUP_SIZE;
        int k = (D + BLOCK_GROUP_SIZE - 1)/BLOCK_GROUP_SIZE + 1;
        tag_info.emplace_back(tag, k, tags[tag].peak_time);
    }

    // 按k降序排序（优先处理大需求标签）
    sort(tag_info.begin(), tag_info.end(), [](const auto& a, const auto& b) {
        return get<1>(a) > get<1>(b);
    });

    // 初始化磁盘信息
    int block_group_num = (V + BLOCK_GROUP_SIZE - 1)/BLOCK_GROUP_SIZE;
    vector<DiskInfo> disk_infos(N+1); // 1-based磁盘信息（自定义结构）
    for (int d = 1; d <= N; ++d) {
        disk_infos[d].free_groups = block_group_num;
        disk_infos[d].peak_times.clear();
    }

    // 步骤2：为每个标签分配磁盘
    for (const auto& elem : tag_info) {
        int tag = get<0>(elem);
        int k = get<1>(elem);
        int current_peak = get<2>(elem);

        vector<int> candidate_disks;
        // 筛选候选磁盘（满足块组数量且峰值时间不冲突）
        for (int d = 1; d <= N; ++d) {
            if (disk_infos[d].free_groups < k) continue;

            // 检查峰值时间冲突
            bool conflict = false;
            for (int pt : disk_infos[d].peak_times) {
                if (abs(pt - current_peak) < PEAK_THRESHOLD) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) candidate_disks.push_back(d);
        }

        // 若严格模式无法满足，放宽条件（允许少量冲突）
        if (candidate_disks.size() < 3) {
            for (int d = 1; d <= N; ++d) {
                if (disk_infos[d].free_groups >= k && 
                    find(candidate_disks.begin(), candidate_disks.end(), d) == candidate_disks.end()
                ) {
                    candidate_disks.push_back(d);
                }
            }
        }

        if (candidate_disks.size() < 3) continue;

        // 排序策略：优先选择 1)冲突最少 2)空闲块组最多
        sort(candidate_disks.begin(), candidate_disks.end(), [&](int a, int b) {
            // 计算当前标签与磁盘已有峰值的冲突数
            auto conflict_count = [&](int disk_id) {
                int cnt = 0;
                for (int pt : disk_infos[disk_id].peak_times) {
                    if (abs(pt - current_peak) < PEAK_THRESHOLD) cnt++;
                }
                return cnt;
            };
            
            if (conflict_count(a) != conflict_count(b)) {
                return conflict_count(a) < conflict_count(b); // 冲突少的优先
            } else {
                return disk_infos[a].free_groups > disk_infos[b].free_groups; // 空闲多的优先
            }
        });

        // 选择最优的三个磁盘
        vector<int> selected_disks = {candidate_disks[0], candidate_disks[1], candidate_disks[2]};

        // 更新磁盘信息
        disk_infos[selected_disks[0]].free_groups -= k;
        disk_infos[selected_disks[0]].peak_times.push_back(current_peak); // 记录峰值时间

        // 标记块组
        Disk& disk = disks[selected_disks[0]];
        int marked = 0;
        for (int bg = 1; bg <= disk.block_group_num && marked < k; ++bg) {
            if (disk.block_groups[bg].selected_type == 0) {
                disk.block_groups[bg].selected_type = tag;
                disk.block_groups[bg].constant = 1;
                marked++;
            }
        }
    
    }
}

void ControlModule::updateRequestTimeouts(int request_size)
{
    req_num += request_size;
    for (int i = req_num; i >= 1; i--)
    {
        if (!requests[i].is_done)
        {
            if (requests[i].value <= 0)
                break;
            requests[i].value -= 0.01f;
        }
    }
}

int Disk::update_hot_tags(ControlModule* cm)
{   
    int start_situation = disk_size;
    float max_value = 0.0f;
    
    auto findReplicaIndex = [&](const Object &obj, int diskId) -> int {
    for (int rep = 1; rep <= REP_NUM; rep++) {
        if (obj.replica[rep] == diskId)
            return rep;
        }
        return -1; // 未找到
    };

    for(int tag=1; tag<=16; tag++){
        int temp_start = disk_size;
        float total_value = 0.0f;
        for(auto rit = tag_hot[tag].rbegin(); rit != tag_hot[tag].rend(); ++rit) {
            Request& req = cm->requests[*rit];
            if( (!req.is_done && req.object_id == 0) || req.req_id == 0 || req.value <= (1.05 - double(HOT_SAMPLE/100.0))) break;
            if (req.object_id == 0) continue;
            total_value += cm->objects[req.object_id].size;
           
            int replicaIndex = findReplicaIndex(cm->objects[req.object_id], id);
            if(cm->objects[req.object_id].unit[replicaIndex][1] < temp_start) temp_start = cm->objects[req.object_id].unit[replicaIndex][1];
        }
        if(total_value > max_value){
            max_value = total_value;
            start_situation = temp_start;
        }

        float value = total_value / HOT_SAMPLE;
        double z_score = (value - cm->tags[tag].var) / cm->tags[tag].std;

        if(z_score >= TAG_THRESHOLD) hot_tags[tag] = 1;
        else hot_tags[tag] = (hot_tags[tag] >= 0 ? hot_tags[tag] - 1 : 0);
    }

    return start_situation;
}

void Disk::update_hot_tags2(ControlModule* cm)
{   
    for(int tag=1; tag<=16; tag++){
        int temp_start = disk_size;
        float total_value = 0.0f;
        for(auto rit = tag_hot[tag].rbegin(); rit != tag_hot[tag].rend(); ++rit) {
            Request& req = cm->requests[*rit];
            if( (!req.is_done && req.object_id == 0) || req.req_id == 0 || req.value <= (1.05 - double(HOT_SAMPLE/100.0))) break;
            if (req.object_id == 0) continue;
            total_value += cm->objects[req.object_id].size;
        }

        float value = total_value / HOT_SAMPLE;
        double z_score = (value - cm->tags[tag].var) / cm->tags[tag].std;

        if(z_score >= TAG_THRESHOLD) hot_tags[tag] = 1;
        else hot_tags[tag] = (hot_tags[tag] >= 0 ? hot_tags[tag] - 1 : 0);
    }
}

void Disk::update_hot_tags3(ControlModule* cm)
{   
    for(int tag=1; tag<=16; tag++){
        int temp_start = disk_size;
        float total_value = 0.0f;
        for(auto rit = tag_hot[tag].rbegin(); rit != tag_hot[tag].rend(); ++rit) {
            Request& req = cm->requests[*rit];
            if( (!req.is_done && req.object_id == 0) || req.req_id == 0 || req.value <= 0.99) break;
            if (req.object_id == 0) continue;
            total_value += cm->objects[req.object_id].size;
        }

        if(total_value >= cm->tags[tag].var) hot_tags[tag] = 50;
        else hot_tags[tag] = (hot_tags[tag] >= 0 ? hot_tags[tag] - 1 : 0);
    }
}

bool allocate_storage(ControlModule* cm, int disk_id, int obj_id, int rep, int from, int to)
{   
    Disk &disk = cm->disks[disk_id];
    Object &obj = cm->objects[obj_id];
    for (int start = from; start <= to - obj.size + 1; start++)
    {
        bool allFree = true;
        for (int j = 0; j < obj.size; j++)
        {
            if (disk.storage[start + j].object_id != 0)
            {
                allFree = false;
                break;
            }
        }
        if (allFree)
        {
            for (int j = 0; j < obj.size; j++)
            {   
                int pos = start + j;
                disk.storage[pos].object_id = obj_id;
                disk.storage[pos].block_id = j + 1;
                obj.unit[rep][j + 1] = pos;
            }
            return true;
        }
    }
    return false;
}

void handle_timestamp(ControlModule *cm)
{
    int timestamp;
    scanf("%*s%d", &timestamp);
    cm->current_time = timestamp;
    printf("TIMESTAMP %d\n", timestamp);
    fflush(stdout);
}

void do_object_delete(const int *object_unit, Disk &disk, int size)
{
    //======================更新 block_group
    int block_group_id = object_unit[1] / BLOCK_GROUP_SIZE;
    if (object_unit[1] % BLOCK_GROUP_SIZE != 0)
    {
        block_group_id++;
    }
    Block_Group &block_group = disk.block_groups[block_group_id];
    block_group.use_size -= size;
    if (block_group.use_size == 0 && block_group.constant != 0)
    {
        block_group.selected_type = 0;
    }
    for (int i = 1; i <= size; i++)
    {
        int unit_index = object_unit[i];
        disk.storage[unit_index].object_id = 0;
        disk.storage[unit_index].block_id = 0;
    }
}

void delete_action(ControlModule *cm)
{
    int n_delete;
    scanf("%d", &n_delete);
    static int _id[MAX_OBJECT_NUM];
    for (int i = 1; i <= n_delete; i++)
    {
        scanf("%d", &_id[i]);
    }

    vector<int> abort_reqs;
    for (int i = 1; i <= n_delete; i++)
    {
        int id = _id[i];
        int current_id = cm->objects[id].last_request_point;
        while (current_id != 0)
        {
            if (!cm->requests[current_id].is_done)
            {
                abort_reqs.push_back(current_id);
                cm->requests[current_id].is_done = true; // 标记为已取消
            }
            current_id = cm->requests[current_id].prev_id;
        }
    }

    printf("%d\n", (int)abort_reqs.size());
    for (int req_id : abort_reqs)
    {
        printf("%d\n", req_id);
    }

    for (int i = 1; i <= n_delete; i++)
    {
        int id = _id[i];
        int current_id = cm->objects[id].last_request_point;
        while (current_id != 0)
        {
            if (!cm->requests[current_id].is_done)
            {
                printf("%d\n", current_id);
            }
            current_id = cm->requests[current_id].prev_id;
        }
        for (int rep = 1; rep <= REP_NUM; rep++)
        {
            int disk_id = cm->objects[id].replica[rep];
            if (disk_id < 1 || disk_id > cm->N)
                continue;
            do_object_delete(cm->objects[id].unit[rep], cm->disks[disk_id], cm->objects[id].size);
        }
        cm->objects[id].is_delete = true;
    }
    fflush(stdout);
}

vector<pair<int, int>> select_block_group(int tag, int size, ControlModule *cm)
{
    vector<pair<int, int>> rep_block_group;
    vector<int> all_disks;
    for (int d = 1; d <= cm->N; d++)
        all_disks.push_back(d);
    vector<int> all_block_groups;
    for (int b = 1; b <= cm->disks[1].block_group_num; b++)
    {
        all_block_groups.push_back(b);
    }
    vector<pair<int, pair<int, int>>> satisfy_block_group;
    for (auto d : all_disks)
    {
        if (rep_block_group.size() >= REP_NUM)
        {
            break;
        }
        int max_b = 0;
        int max_size = 0;
        for (auto b : all_block_groups)
        {
            if (cm->disks[d].block_groups[b].selected_type != tag) continue;
            int start = cm->disks[d].block_groups[b].start_pos;
            int end = cm->disks[d].block_groups[b].end_pos;
            bool hasContiguous = false;
            for (int j = start; j <= end - size + 1; j++)
            {
                bool contiguous = true;
                for (int k = 0; k < size; k++)
                {
                    if (cm->disks[d].storage[j + k].object_id != 0)
                    {
                        contiguous = false;
                        break;
                    }
                }
                if (contiguous)
                {
                    hasContiguous = true;
                    break;
                }
            }
            if (hasContiguous)
            {
                if (cm->disks[d].block_groups[b].use_size > max_size)
                {
                    max_size = cm->disks[d].block_groups[b].use_size;
                    max_b = b;
                }
            }
        }
        if (max_b != 0)
        {   
            satisfy_block_group.push_back(make_pair(d, make_pair(max_b, max_size)));
        }
    }
    std::sort(satisfy_block_group.begin(), satisfy_block_group.end(), [](const auto &a, const auto &b)
              { return a.second.second > b.second.second; });

    for (auto &group : satisfy_block_group)
    {
        if (rep_block_group.size() >= REP_NUM) break;
        rep_block_group.push_back(make_pair(group.first, group.second.first));
    }
    satisfy_block_group.clear();

    for (auto d : all_disks)
    {
        if (rep_block_group.size() >= REP_NUM)
        {
            break;
        }
        int min_b = 0;
        int min_size = BLOCK_GROUP_SIZE;
        bool repeat = false;
        for (auto &rep_disk_id : rep_block_group)
        {
            if (rep_disk_id.first == d)
            {
                repeat = true;
                break;
            }
        }
        if (repeat)
        {
            continue;
        }
        for (auto b : all_block_groups)
        {
            int start = cm->disks[d].block_groups[b].start_pos;
            int end = cm->disks[d].block_groups[b].end_pos;
            bool hasContiguous = false;
            for (int j = start; j <= end - size + 1; j++)
            {
                bool contiguous = true;
                for (int k = 0; k < size; k++)
                {
                    if (cm->disks[d].storage[j + k].object_id != 0)
                    {
                        contiguous = false;
                        break;
                    }
                }
                if (contiguous)
                {
                    hasContiguous = true;
                    break;
                }
            }
            if (hasContiguous)
            {
                if (cm->disks[d].block_groups[b].use_size < min_size)
                {
                    min_size = cm->disks[d].block_groups[b].use_size;
                    min_b = b;
                }
            }
        }
        if (min_b != 0)
        {
            satisfy_block_group.push_back(make_pair(d, make_pair(min_b, min_size)));
        }
    }

    std::sort(satisfy_block_group.begin(), satisfy_block_group.end(), [](const auto &a, const auto &b)
              { return a.second.second < b.second.second; });

    for (auto &group : satisfy_block_group)
    {
        if (rep_block_group.size() >= REP_NUM)
        {
            break;
        }
        rep_block_group.push_back(make_pair(group.first, group.second.first));
    }
    return rep_block_group;
}

void write_action(ControlModule *cm)
{
    int n_write;
    scanf("%d", &n_write);
    for (int i = 0; i < n_write; i++)
    {
        int obj_id, size, tag;
        scanf("%d%d%d", &obj_id, &size, &tag);
        if (obj_id < 1 || obj_id >= MAX_OBJECT_NUM)
            continue;

        Object &obj = cm->objects[obj_id];
        obj.obj_id = obj_id;
        obj.size = size;
        obj.tag = tag;
        obj.last_request_point = 0;
        obj.is_delete = false;

        if (obj.block_read_status)
            delete[] obj.block_read_status;
        obj.block_read_status = new bool[size + 1]();
        vector<pair<int, int>> rep_block_group = select_block_group(tag, obj.size, cm);
        for (int rep = 1; rep <= REP_NUM; rep++)
        {
            if (obj.unit[rep])
                delete[] obj.unit[rep];
            obj.unit[rep] = new int[size + 1];

            Disk &disk = cm->disks[rep_block_group[rep - 1].first];
            Block_Group &block_group = disk.block_groups[rep_block_group[rep - 1].second];
            bool ok = allocate_storage(cm, rep_block_group[rep - 1].first, obj_id, rep, block_group.start_pos, block_group.end_pos);
            assert(ok);
            obj.replica[rep] = rep_block_group[rep - 1].first;
            if (block_group.selected_type == 0)
            {
                block_group.selected_type = tag;
            }
            block_group.use_size += size;
        }
        printf("%d\n", obj_id);
        for (int rep = 1; rep <= REP_NUM; rep++)
        {
            printf("%d", obj.replica[rep]);
            for (int b = 1; b <= size; b++)
            {
                printf(" %d", obj.unit[rep][b]);
            }
            printf("\n");
        }
    }

    fflush(stdout);
}

void CheckAndCompleteRequests(ControlModule* cm, int start_req_id, vector<int>& completed) {
    int current_req = start_req_id;
    Request& req = cm->requests[current_req];

    while (req.value > 0 && !req.is_done) { // 要一直检查到有读的请求，或者价值为 0 的请求，也就是超时请求
        Object& obj = cm->objects[req.object_id];
        bool all_blocks_read = true;
        
        for(int block=1; block<=obj.size; ++block){
            if(!req.block_read_status[block]){
                all_blocks_read = false;
                // break; // 发现未读块不要退出啊，说不定前面一个请求有读呢？要继续检查
            }
        }

        if (all_blocks_read) {
            req.value = 0;
            req.is_done = true;
            completed.push_back(current_req);
            cm->tags[obj.tag].score += (req.value + 0.05f);
        }
        current_req = req.prev_id;
    }
}

void read_action(ControlModule *cm)
{
    int n_read;
    scanf("%d", &n_read);

    for (int i = 0; i < n_read; i++)
    {
        int req_id, obj_id;
        scanf("%d%d", &req_id, &obj_id);
        Request &req = cm->requests[req_id];
        Object &obj = cm->objects[obj_id];
        req.req_id = req_id;
        req.is_done = false;
        req.object_id = obj_id;
        req.arrival_time = cm->current_time;
        req.value = 1.05f;
        if (req.block_read_status)
            delete[] req.block_read_status;
        req.block_read_status = new bool[obj.size + 1]();

        for (int rep = 1; rep <= REP_NUM; rep++)
        {
            cm->disks[obj.replica[rep]].tag_hot[obj.tag].push_back(req_id);
        }

        req.prev_id = cm->objects[obj_id].last_request_point;
        cm->objects[obj_id].last_request_point = req_id;
    }

    cm->updateRequestTimeouts(n_read);

    vector<int> completed;

    for (int d = 1; d <= cm->N; d++)
    {
        Disk &disk = cm->disks[d];
        int hot_situation = disk.update_hot_tags(cm);
        // disk.update_hot_tags2(cm);
        string instruction;
        disk.used_tokens = 0;
        

        while (cm->G - disk.used_tokens > 0)
        {
            Object &head_obj = cm->objects[disk.storage[disk.head_position].object_id];
            if (cm->requests[head_obj.last_request_point].value > 0 && head_obj.obj_id != 0)
            {
                Object &obj = cm->objects[disk.storage[disk.head_position].object_id];
                int read_token = cm->nextTokenLookup[disk.last_token_cost];
                if (cm->G - disk.used_tokens > read_token)
                {
                    instruction += 'r';
                    disk.used_tokens += read_token;
                    disk.last_token_cost = read_token;

                    int current_req = obj.last_request_point;
                    // CheckAndCompleteRequests(cm, current_req, completed);
                    while (cm->requests[current_req].value > 0 && !cm->requests[current_req].is_done)
                    {
                        int block_id = disk.storage[disk.head_position].block_id;
                        cm->requests[current_req].block_read_status[block_id] = true;
                        bool req_statu = true;
                        for (int block = 1; block <= obj.size; block++)
                        {
                            if (!cm->requests[current_req].block_read_status[block])
                            {
                                req_statu = false;
                                break;
                            }
                        }
                        if (req_statu)
                        {   
                            cm->tags[obj.tag].score += (cm->requests[current_req].value);
                            cm->requests[current_req].value = 0;      // value 置零
                            cm->requests[current_req].is_done = true; // 标记已读
                            completed.push_back(current_req);         // 加入缓存
                        }
                        current_req = cm->requests[current_req].prev_id; // 检查上一个请求
                    }
                    disk.head_position = (disk.head_position % cm->V) + 1;
                }
                else
                {
                    instruction += '#';
                    break;
                }
            }
            // 情况2：当前位置无任务，寻找下一个任务位置
            else
            {
                int distance = 0, j;
                bool has_hot_tag = false;
                for (int tag = 1; tag <= cm->M; tag++) {
                    if (disk.hot_tags[tag] > 0) {
                        has_hot_tag = true;
                        break;
                    }
                }

                for (j = disk.head_position % cm->V + 1; j != disk.head_position; j = (j % cm->V) + 1)
                {
                    ++distance; // 计算移动步数
                    Object &obj = cm->objects[disk.storage[j].object_id];
                    if (has_hot_tag) {
                    if (disk.hot_tags[obj.tag] > 0 && cm->requests[obj.last_request_point].value > LOW_SCORE_THRESHOLD) {
                        break;
                    }
                    } else {
                        // 原条件：无热标签时按 value 判断
                        if (cm->requests[obj.last_request_point].value > 0.78) {
                            break;
                        }
                    }
                }
                if (j == disk.head_position)
                {   
                    instruction += '#';
                    break;
                }
                // 情况2.1：移动距离超过最大令牌数，且令牌未被使用过

                if (distance >= (cm->G - 64))
                {   
                    int jump;
                    if(hot_situation != disk.disk_size) jump = hot_situation;
                    else jump = j;
                    if (disk.used_tokens == 0)
                    {                                     
                        instruction = "j ";               // 输出Jump指令
                        instruction += std::to_string(jump); // 将整数j转换为字符串并拼接
                        disk.head_position = jump;
                        disk.last_token_cost = cm->G;
                        break;
                    }
                }

                if (distance <= SHORT_DISTANCE){
                    // debug_function(cm, 9, distance, 987, cm->G - disk.used_tokens);
                    Object& obj = cm->objects[disk.storage[j].object_id];
                    auto key = std::make_tuple(distance, disk.last_token_cost, obj.size);
                    bool strategy = cm->tokenStrategyLookup[key];
                    if(strategy){
                        // debug_function(cm, 0, obj.last_request_point, obj.size, obj.obj_id);
                        Object& obj = cm->objects[disk.storage[disk.head_position].object_id];
                        int read_token = cm->nextTokenLookup[disk.last_token_cost];
                        if(cm->G - disk.used_tokens > read_token){
                            instruction += 'r';
                            disk.used_tokens += read_token;
                            disk.last_token_cost = read_token;
                            // debug_function(cm, 9, d, 985, cm->G - disk.used_tokens);

                            int current_req = obj.last_request_point;
                            while(cm->requests[current_req].value > 0 && !cm->requests[current_req].is_done){
                                int block_id = disk.storage[disk.head_position].block_id;
                                cm->requests[current_req].block_read_status[block_id] = true;
                                bool req_statu = true;
                                for (int block = 1; block <= obj.size; block++) {
                                    if (!cm->requests[current_req].block_read_status[block]) {
                                        req_statu = false;
                                        break;
                                    }
                                }
                                if(req_statu){
                                    cm->tags[obj.tag].score += (cm->requests[current_req].value + 0.05);
                                    cm->requests[current_req].value = 0;            // value 置零
                                    cm->requests[current_req].is_done = true;       // 标记已读
                                    completed.push_back(current_req);               // 加入缓存
                                }
                                current_req = cm->requests[current_req].prev_id;    // 检查上一个请求
                                }
                            disk.head_position = (disk.head_position % cm->V) + 1;
                        } else{
                            instruction += '#';
                            break;
                        }
                        continue;
                    }
                    else{
                        distance = min(distance, cm->G - disk.used_tokens); // 实际可移动步数
                        for (; distance > 0;)
                        {
                            instruction += 'p'; // 输出Jump指令
                            disk.last_token_cost = 1;
                            disk.head_position = (disk.head_position % cm->V) + 1; // 移动磁头
                            disk.used_tokens++;
                            --distance;
                        }

                        if (disk.used_tokens == cm->G)
                        {
                            instruction += '#';
                            break;
                        }
                        continue;
                    }
                }

                if(d == 1) debug_function(cm, 9, d, distance, 964);

                // 情况2.2：移动并消耗令牌
                distance = min(distance, cm->G - disk.used_tokens); // 实际可移动步数
                for (; distance > 0;)
                {
                    instruction += 'p'; // 输出Jump指令
                    disk.last_token_cost = 1;
                    disk.head_position = (disk.head_position % cm->V) + 1; // 移动磁头
                    disk.used_tokens++;
                    --distance;
                }

                // 令牌用完则结束
                if (disk.used_tokens == cm->G)
                {
                    instruction += '#';
                    break;
                }
            }
        }
        // printf("%s\n", instruction.c_str());

        printf("%s\n", instruction.c_str());
    }

    // 输出完成请求
    printf("%d\n", (int)completed.size());
    for (int id : completed)
    {
        printf("%d\n", id);
    }
    fflush(stdout);

    // if(cm->current_time % 10000 == 0){
    // for (int i = 1; i <= cm->M; i++)
    //     {
    //     debug_function(cm, 0, i, cm->tags[i].score, 1);
    //     }
    // }

}


void clean(ControlModule *cm)
{
    fflush(stdout);
}

// ---------------------------
// 主函数
// ---------------------------
int main() {
    ControlModule* cm = new ControlModule();

    scanf("%d%d%d%d%d", &cm->T, &cm->M, &cm->N, &cm->V, &cm->G);

    int slices = (cm->T - 1) / FRE_PER_SLICING + 1;

    for (int i = 1; i <= cm->M; i++){
        cm->tags[i].delete_obj= new int[slices+1];
        cm->tags[i].write_obj = new int[slices+1];
        cm->tags[i].read_obj = new int[slices+1];
    }

    for (int i = 0; i < cm->M * 3; i++) {
        for (int j = 1; j <= slices; j++) {
            int val;
            scanf("%d", &val);
            if (i < cm->M) { 
                int tag_id = i + 1;
                cm->tags[tag_id].delete_obj[j] = val;
            } else if (i < 2*cm->M) { 
                int tag_id = (i - cm->M) + 1;
                cm->tags[tag_id].write_obj[j] = val;
            } else {
                int tag_id = (i - 2*cm->M) + 1;
                cm->tags[tag_id].read_obj[j] = val;
            }
        }
    }

    printf("OK\n");
    fflush(stdout);

    //=======================初始化block_group_num
    int block_group_num = cm->V / BLOCK_GROUP_SIZE;
    if (cm->V % BLOCK_GROUP_SIZE != 0)
    {
        block_group_num++;
    }
    // 初始化硬盘状态（1-based索引）
    for (int d = 1; d <= cm->N; d++)
    {   
        Disk& disk = cm->disks[d];
        disk.id = d;
        disk.head_position = 1;
        disk.last_token_cost = 0;
        disk.used_tokens = 0;
        disk.disk_size = cm->V;
        for (int j = 1; j <= cm->V; j++)
        {
            disk.storage[j].object_id = 0;
            disk.storage[j].block_id = 0;
        }
        disk.initBlockGroup(block_group_num, cm->V);
        for (int j = 1; j <= cm->M; j++)
        {
            disk.tag_hot[j].push_back(0);
        }
    }

    // 初始化对象和请求（数组下标0未用）
    for (int i = 0; i < MAX_OBJECT_NUM; i++)
    {
        cm->objects[i].size = 0;
        cm->objects[i].tag = 0;
        cm->objects[i].last_request_point = 0;
        cm->objects[i].is_delete = false;
        for (int rep = 1; rep <= REP_NUM; rep++)
        {
            cm->objects[i].unit[rep] = nullptr;
        }
        cm->objects[i].block_read_status = nullptr;
    }
    for (int i = 0; i < MAX_REQUEST_NUM; i++)
    {
        cm->requests[i].is_done = false;
    }

    cm->InitTokenLookupTables();
    cm->AssignPrimaryDisks();

    // 模拟 T + EXTRA_TIME 个时间片事件
    // debug_function(cm, 0, 123654, 10, cm->current_time);

    for (int t = 1; t <= cm->T + EXTRA_TIME; t++)
    {
        handle_timestamp(cm);
        delete_action(cm);
        write_action(cm);
        read_action(cm);
        // if(cm->current_time == 200){
        //     debug_function(cm, 0, cm->max_distance, cm->max_distance, cm->max_distance);
        // }
    }

    clean(cm);
    return 0;
}
