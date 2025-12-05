package com.example.chat.mapper;


import com.example.chat.domain.dto.chat.request.CreateChatRequest;
import com.example.chat.domain.dto.chat.response.ChatResponse;
import com.example.chat.domain.dto.chat.response.ChatWithMessagesResponse;
import com.example.chat.domain.dto.chat.response.CreateChatResponse;
import com.example.chat.domain.entities.Chat;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;

@Mapper(componentModel = "spring", unmappedTargetPolicy = ReportingPolicy.IGNORE)
public interface ChatMapper {
    Chat toEntity(CreateChatRequest chatDTO);

    CreateChatResponse toCreateChatResponse(Chat chat);
    ChatWithMessagesResponse toChatWithMessagesResponse(Chat chat);
    ChatResponse toChatResponse(Chat chat);
}
