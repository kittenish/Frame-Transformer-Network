%% random emotion6
%avg : joy : 3230 / 4096 = 0
%most: 0-10
% function random_emotion6(opt,url)
%     feats = load(url);
%     feats = feats.feats;
%     label = regexp(url,'/','split');
%     label = label{7};
%     data = zeros(500, opt, 4096);
%     position = zeros(500, 2);
%     
%     for i = 1:500
%         i
%         [data(i,:,:), position(i,:)] = random_data(opt,feats);
%     end
%     
%     save('joy_data_500_fc7.mat', 'data');
%     save('joy_position_500_fc7.mat', 'position');
% end
% 
% function [data,position] = random_data(opt,feats)
%     data = zeros(opt, 4096);
%     senti_len = randi(5, 1) + 5;
%     
%     position = zeros(1, 2);
%     begin = randi(opt - senti_len - 1, 1);
%     stop = begin + senti_len - 1;
%     position(1,1) = begin;
%     position(1,2) = stop;
%     
%     for i = 1:opt
%         if i >= begin && i <= stop
%             j = randi(330, 1);
%             data(i,:) = feats(j,:);
%         else
%             for j = 1:4096
%                 m = randi(4, 1);
%                 if m / 4 == 1
%                     data(i,j) = rand(1) * 10;
%                 end
%             end
% %             random_num = rand(1,1000) * 15;
% %             data(i,:) = random_num;
%         end
%     end
% end

%% random emotion6 distort
%joy data: joy:12-20; anger: 3-5
%anger data: anger:12-20; joy: 3-5
function [train_data, train_position, test_data, test_position, validation_data, validation_position] = random_emotion6(opt,url,anger_feats, joy_feats, disgust_feats, sadness_feats, surprise_feats, fear_feats)
    feats = load(url);
    feats = feats.feats;
    l = {'anger', 'fear', 'joy', 'disgust', 'surprise', 'sadness'};
    black_feats = load('/Volumes/Transcend/mat/black_white/black_fc7_100.mat');
    black = black_feats.feats;
    white_feats = load('/Volumes/Transcend/mat/black_white/white_fc7_100.mat');
    white = white_feats.feats;
    train_data = zeros(400, opt, 4096);
    train_position = zeros(400, 2);
    test_data = zeros(100, opt, 4096);
    test_position = zeros(100, 2);
    validation_data = zeros(100, opt, 4096);
    validation_position = zeros(100, 2);
    
    for i = 1:400
        i
        [train_data(i,:,:), train_position(i,:)] = random_data_train(l,opt,feats,black,white,anger_feats, joy_feats, disgust_feats, sadness_feats, surprise_feats, fear_feats);
    end
    
    for i = 1:100
        i
        [test_data(i,:,:), test_position(i,:)] = random_data_test(l,opt,feats,black,white,anger_feats, joy_feats, disgust_feats, sadness_feats, surprise_feats, fear_feats);
        [validation_data(i,:,:), validation_position(i,:)] = random_data_validation(l,opt,feats,black,white,anger_feats, joy_feats, disgust_feats, sadness_feats, surprise_feats, fear_feats);
    end
    
end

function [data,position,position_dis] = random_data_train(l,opt,feats,black,white,anger_feats, joy_feats, disgust_feats, sadness_feats, surprise_feats, fear_feats)
    data = zeros(opt, 4096);
    senti_len = randi(8, 1) + 12;
    dis_len = randi(3, 1) + 2;
    tmp = randi(2, 1);
    
    if tmp == 1
        position = zeros(1, 2);
        begin = randi(opt - senti_len - dis_len - 1, 1);
        stop = begin + senti_len - 1;
        position(1,1) = begin;
        position(1,2) = stop;
        
        position_dis = zeros(1, 2);
        begin_dis = randi(opt - stop - 1 - dis_len, 1) + stop;
        stop_dis = begin_dis + dis_len - 1;
        position_dis(1,1) = begin_dis;
        position_dis(1,2) = stop_dis;
    else
        position_dis = zeros(1, 2);
        begin_dis = randi(opt - senti_len - dis_len - 1, 1);
        stop_dis = begin_dis + dis_len - 1;
        position_dis(1,1) = begin_dis;
        position_dis(1,2) = stop_dis;
        
        position = zeros(1, 2);
        begin = randi(opt - stop_dis - 1 - senti_len, 1) + stop_dis;
        stop = begin + senti_len - 1;
        position(1,1) = begin;
        position(1,2) = stop;
    end
        
    for i = 1:opt
        if i >= begin && i <= stop
            j = randi(210, 1);
            data(i,:) = feats(j,:);
        elseif i >= begin_dis && i <= stop_dis
            color = randi(2,1);
            if color == 1
                j = randi(60, 1);
                data(i,:) = black(j,:);
            else
                j = randi(60, 1);
                data(i,:) = white(j,:);
            end
        else
            j = randi(210, 1);
            temp = randi(6, 1);
            c = char(l{temp});
            c = eval([c,'_feats']);
            data(i,:) = c(j,:);
        end
    end
end

function [data,position,position_dis] = random_data_test(l,opt,feats,black,white,anger_feats, joy_feats, disgust_feats, sadness_feats, surprise_feats, fear_feats)
    data = zeros(opt, 4096);
    senti_len = randi(8, 1) + 12;
    dis_len = randi(3, 1) + 2;
    tmp = randi(2, 1);
    
    if tmp == 1
        position = zeros(1, 2);
        begin = randi(opt - senti_len - dis_len - 1, 1);
        stop = begin + senti_len - 1;
        position(1,1) = begin;
        position(1,2) = stop;
        
        position_dis = zeros(1, 2);
        begin_dis = randi(opt - stop - 1 - dis_len, 1) + stop;
        stop_dis = begin_dis + dis_len - 1;
        position_dis(1,1) = begin_dis;
        position_dis(1,2) = stop_dis;
    else
        position_dis = zeros(1, 2);
        begin_dis = randi(opt - senti_len - dis_len - 1, 1);
        stop_dis = begin_dis + dis_len - 1;
        position_dis(1,1) = begin_dis;
        position_dis(1,2) = stop_dis;
        
        position = zeros(1, 2);
        begin = randi(opt - stop_dis - 1 - senti_len, 1) + stop_dis;
        stop = begin + senti_len - 1;
        position(1,1) = begin;
        position(1,2) = stop;
    end
        
    for i = 1:opt
        if i >= begin && i <= stop
            j = randi(60, 1) + 210;
            data(i,:) = feats(j,:);
        elseif i >= begin_dis && i <= stop_dis
            color = randi(2,1);
            if color == 1
                j = randi(20, 1) + 60;
                data(i,:) = black(j,:);
            else
                j = randi(20, 1) + 60;
                data(i,:) = white(j,:);
            end
        else
            j = randi(60, 1) + 210;
            temp = randi(6, 1);
            c = char(l{temp});
            c = eval([c,'_feats']);
            data(i,:) = c(j,:);
        end
    end
end

function [data,position,position_dis] = random_data_validation(l,opt,feats,black,white,anger_feats, joy_feats, disgust_feats, sadness_feats, surprise_feats, fear_feats)
    data = zeros(opt, 4096);
    senti_len = randi(8, 1) + 12;
    dis_len = randi(3, 1) + 2;
    tmp = randi(2, 1);
    
    if tmp == 1
        position = zeros(1, 2);
        begin = randi(opt - senti_len - dis_len - 1, 1);
        stop = begin + senti_len - 1;
        position(1,1) = begin;
        position(1,2) = stop;
        
        position_dis = zeros(1, 2);
        begin_dis = randi(opt - stop - 1 - dis_len, 1) + stop;
        stop_dis = begin_dis + dis_len - 1;
        position_dis(1,1) = begin_dis;
        position_dis(1,2) = stop_dis;
    else
        position_dis = zeros(1, 2);
        begin_dis = randi(opt - senti_len - dis_len - 1, 1);
        stop_dis = begin_dis + dis_len - 1;
        position_dis(1,1) = begin_dis;
        position_dis(1,2) = stop_dis;
        
        position = zeros(1, 2);
        begin = randi(opt - stop_dis - 1 - senti_len, 1) + stop_dis;
        stop = begin + senti_len - 1;
        position(1,1) = begin;
        position(1,2) = stop;
    end
        
    for i = 1:opt
        if i >= begin && i <= stop
            j = randi(60, 1) + 270;
            data(i,:) = feats(j,:);
        elseif i >= begin_dis && i <= stop_dis
            color = randi(2,1);
            if color == 1
                j = randi(20, 1) + 80;
                data(i,:) = black(j,:);
            else
                j = randi(20, 1) + 80;
                data(i,:) = white(j,:);
            end
        else
            j = randi(60, 1) + 270;
            temp = randi(6, 1);
            c = char(l{temp});
            c = eval([c,'_feats']);
            data(i,:) = c(j,:);
        end
    end
end


%% random emotion6 fc7 distort all by color
% %joy data: joy:10-20; other: black / white
% %anger data: anger:10-20; other: black / white
% function random_emotion6(opt,url)
%     feats = load(url);
%     feats = feats.feats;
%     black_feats = load('/Users/mac/Desktop/video_spacial/mat/black_white/black.mat');
%     black = black_feats.feats;
%     white_feats = load('/Users/mac/Desktop/video_spacial/mat/black_white/white.mat');
%     white = white_feats.feats;
%     label = regexp(url,'/','split');
%     label = label{7};
%     data = zeros(500, opt, 4096);
%     position = zeros(500, 2);
%     
%     for i = 1:500
%         i
%         [data(i,:,:), position(i,:)] = random_data(opt,feats,black,white);
%     end
%     
%     save('./mat/emotion6/all_color_joy_data_500_fc7.mat', 'data');
%     save('./mat/emotion6/all_color_joy_position_500_fc7.mat', 'position');
% end
% 
% function [data,position] = random_data(opt,feats,black,white)
%     data = zeros(opt, 4096);
%     senti_len = randi(10, 1) + 10;
%     tmp_begin = randi(2, 1);
%     tmp_end = randi(2, 1);
%     
%     position = zeros(1, 2);
%     begin = randi(opt - senti_len - 1, 1);
%     stop = begin + senti_len - 1;
%     position(1,1) = begin;
%     position(1,2) = stop;
%        
%     for i = 1:opt
%         if i >= begin && i <= stop
%             j = randi(330, 1);
%             data(i,:) = feats(j,:);
%         elseif i >= stop  
%             if tmp_end == 1
%                 j = randi(36, 1);
%                 data(i,:) = black(j,:);
%             else
%                 j = randi(34, 1);
%                 data(i,:) = white(j,:);
%             end
%         else
%             if tmp_begin == 1
%                 j = randi(36, 1);
%                 data(i,:) = black(j,:);
%             else
%                 j = randi(34, 1);
%                 data(i,:) = white(j,:);
%             end
%         end
%     end
% end

%% random emotion6 conv5 distort all by color
% %joy data: joy:10-20; other: black / white
% %anger data: anger:10-20; other: black / white
% function random_emotion6(opt,url)
%     feats = load(url);
%     feats = feats.feats;
%     black_feats = load('/Volumes/Transcend/mat/black_white/black_conv5.mat');
%     black = black_feats.feats;
%     white_feats = load('/Volumes/Transcend/mat/black_white/white_conv5.mat');
%     white = white_feats.feats;
%     label = regexp(url,'/','split');
%     label = label{7};
%     data = zeros(500, opt, 43264);
%     position = zeros(500, 2);
%     
%     for i = 1:500
%         i
%         [data(i,:,:), position(i,:)] = random_data(opt,feats,black,white);
%     end
%     
%     save('/Volumes/Transcend/mat/emotion6/all_color_joy_data_500_conv5.mat', 'data');
%     save('/Volumes/Transcend/mat/emotion6/all_color_joy_position_500_conv5.mat', 'position');
% end
% 
% function [data,position] = random_data(opt,feats,black,white)
%     data = zeros(opt, 43264);
%     senti_len = randi(10, 1) + 10;
%     tmp_begin = randi(2, 1);
%     tmp_end = randi(2, 1);
%     
%     position = zeros(1, 2);
%     begin = randi(opt - senti_len - 1, 1);
%     stop = begin + senti_len - 1;
%     position(1,1) = begin;
%     position(1,2) = stop;
%        
%     for i = 1:opt
%         if i >= begin && i <= stop
%             j = randi(330, 1);
%             data(i,:) = feats(j,:);
%         elseif i >= stop  
%             if tmp_end == 1
%                 j = randi(36, 1);
%                 data(i,:) = black(j,:);
%             else
%                 j = randi(34, 1);
%                 data(i,:) = white(j,:);
%             end
%         else
%             if tmp_begin == 1
%                 j = randi(36, 1);
%                 data(i,:) = black(j,:);
%             else
%                 j = randi(34, 1);
%                 data(i,:) = white(j,:);
%             end
%         end
%     end
% end

%% random all classes in emotion6 fc7 distort by color
%joy data: joy:10-20; other: black / white
%anger data: anger:10-20; other: black / white
% function [train_data, train_position, test_data, test_position, validation_data, validation_position] = random_emotion6(opt,url)
%     feats = load(url);
%     feats = feats.feats;
%     black_feats = load('/Volumes/Transcend/mat/black_white/black_fc7_100.mat');
%     black = black_feats.feats;
%     white_feats = load('/Volumes/Transcend/mat/black_white/white_fc7_100.mat');
%     white = white_feats.feats;
%     label = regexp(url,'/','split');
%     label = label{7};
%     train_data = zeros(400, opt, 4096);
%     train_position = zeros(400, 2);
%     test_data = zeros(100, opt, 4096);
%     test_position = zeros(100, 2);
%     validation_data = zeros(100, opt, 4096);
%     validation_position = zeros(100, 2);
%     
%     for i = 1:400
%         i
%         [train_data(i,:,:), train_position(i,:)] = random_data_train(opt,feats,black,white);
%     end
%     
%     for i = 1:100
%         i
%         [test_data(i,:,:), test_position(i,:)] = random_data_test(opt,feats,black,white);
%         [validation_data(i,:,:), validation_position(i,:)] = random_data_validation(opt,feats,black,white);
%     end
%     
%     %save('./mat/emotion6/all_color_joy_data_500_fc7.mat', 'data');
%     %save('./mat/emotion6/all_color_joy_position_500_fc7.mat', 'position');
% end
% 
% function [data,position] = random_data_train(opt,feats,black,white)
%     data = zeros(opt, 4096);
%     senti_len = randi(10, 1) + 10;
%     tmp_begin = randi(2, 1);
%     tmp_end = randi(2, 1);
%     
%     position = zeros(1, 2);
%     begin = randi(opt - senti_len - 1, 1);
%     stop = begin + senti_len - 1;
%     position(1,1) = begin;
%     position(1,2) = stop;
%        
%     for i = 1:opt
%         if i >= begin && i <= stop
%             j = randi(210, 1);
%             data(i,:) = feats(j,:);
%         elseif i >= stop  
%             if tmp_end == 1
%                 j = randi(60, 1);
%                 data(i,:) = black(j,:);
%             else
%                 j = randi(60, 1);
%                 data(i,:) = white(j,:);
%             end
%         else
%             if tmp_begin == 1
%                 j = randi(60, 1);
%                 data(i,:) = black(j,:);
%             else
%                 j = randi(60, 1);
%                 data(i,:) = white(j,:);
%             end
%         end
%     end
% end
% 
% function [data,position] = random_data_test(opt,feats,black,white)
%     data = zeros(opt, 4096);
%     senti_len = randi(10, 1) + 10;
%     tmp_begin = randi(2, 1);
%     tmp_end = randi(2, 1);
%     
%     position = zeros(1, 2);
%     begin = randi(opt - senti_len - 1, 1);
%     stop = begin + senti_len - 1;
%     position(1,1) = begin;
%     position(1,2) = stop;
%        
%     for i = 1:opt
%         if i >= begin && i <= stop
%             j = randi(60, 1) + 210;
%             data(i,:) = feats(j,:);
%         elseif i >= stop  
%             if tmp_end == 1
%                 j = randi(20, 1) + 60;
%                 data(i,:) = black(j,:);
%             else
%                 j = randi(20, 1) + 60;
%                 data(i,:) = white(j,:);
%             end
%         else
%             if tmp_begin == 1
%                 j = randi(20, 1) + 60;
%                 data(i,:) = black(j,:);
%             else
%                 j = randi(20, 1) + 60;
%                 data(i,:) = white(j,:);
%             end
%         end
%     end
% end
% 
% function [data,position] = random_data_validation(opt,feats,black,white)
%     data = zeros(opt, 4096);
%     senti_len = randi(10, 1) + 10;
%     tmp_begin = randi(2, 1);
%     tmp_end = randi(2, 1);
%     
%     position = zeros(1, 2);
%     begin = randi(opt - senti_len - 1, 1);
%     stop = begin + senti_len - 1;
%     position(1,1) = begin;
%     position(1,2) = stop;
%        
%     for i = 1:opt
%         if i >= begin && i <= stop
%             j = randi(60, 1) + 270;
%             data(i,:) = feats(j,:);
%         elseif i >= stop  
%             if tmp_end == 1
%                 j = randi(20, 1) + 80;
%                 data(i,:) = black(j,:);
%             else
%                 j = randi(20, 1) + 80;
%                 data(i,:) = white(j,:);
%             end
%         else
%             if tmp_begin == 1
%                 j = randi(20, 1) + 80;
%                 data(i,:) = black(j,:);
%             else
%                 j = randi(20, 1) + 80;
%                 data(i,:) = white(j,:);
%             end
%         end
%     end
% end
